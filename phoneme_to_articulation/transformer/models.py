import math
import torch
import torch.nn as nn

from helpers import make_padding_mask
from phoneme_to_articulation.encoder_decoder.models import ArticulatorPredictor


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0,
        max_len: int = 5000
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Torch tensor of shape (bs, seq_len, embedding_dim)
        """
        _, max_length, _ = x.shape

        x = x + self.pe[:, :max_length, :]
        return self.dropout(x)


class ChannelProcessingLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        batch_first: bool = False,
    ):
        super().__init__()

        self.query = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )

        self.key = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )

        self.value = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
        )

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        src,
        tgt,
        key_padding_mask,
        is_causal=False,
        attn_mask=None
    ):
        src = self.layer_norm(src)
        tgt = self.layer_norm(tgt)

        query = self.query(tgt)
        key = self.key(src)
        value = self.value(src)

        out, attn_weights = self.multihead_attn(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            attn_mask=attn_mask,
        )

        # out : (bs, seq_len, embed_dim)
        # attn_weights : (bs, seq_len, seq_len)

        out = query + out

        return out


class ChannelInteractionsLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_tgt_channels: int,
        dropout: float = 0.0,
        batch_first: bool = False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.interactions = nn.ModuleList([
            ChannelProcessingLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=batch_first,
            ) for _ in range(num_tgt_channels)
        ])

        self.num_tgt_channels = num_tgt_channels

        self.linear = nn.Sequential(
            nn.LayerNorm(num_tgt_channels * embed_dim),
            nn.Linear(num_tgt_channels * embed_dim, embed_dim),
            nn.ReLU()
        )

    def forward(
        self,
        src_channel,
        tgt_channels,
        key_padding_mask,
        is_causal=False,
        attn_mask=None,
    ):
        """
        Args:
            src_channel (torch.tensor): Tensor of shape (bs, seq_len, num_feat)
            tgt_channel (torch.tensor): Tensor of shape (bs, num_tgt_channels, seq_len, num_feat)
        """
        outputs = []
        for i in range(self.num_tgt_channels):
            src = self.dropout(src_channel)
            tgt = self.dropout(tgt_channels[:, i, :, :])

            out = self.interactions[i](
                src,
                tgt,
                key_padding_mask,
                is_causal=is_causal,
                attn_mask=attn_mask,
            )  # (bs, seq_len, num_feat)
            outputs.append(out)

        output = self.dropout(torch.cat(outputs, dim=2))
        output = self.linear(output)

        return output


class MultiChannelTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        nchan: int,
        dropout: float = 0,
        batch_first: bool = False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.chan_processing_layers = nn.ModuleList([
            ChannelProcessingLayer(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=batch_first,
            ) for _ in range(nchan)
        ])

        self.chan_interaction_layers = nn.ModuleList([
            ChannelInteractionsLayer(
                embed_dim=d_model,
                num_heads=nhead,
                num_tgt_channels=nchan-1,
                dropout=dropout,
                batch_first=batch_first,
            ) for _ in range(nchan)
        ])

        self.chan_input_layers = nn.ModuleList([
            ChannelProcessingLayer(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=batch_first,
            ) for _ in range(nchan)
        ])

        self.feed_forward = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.num_channels = nchan
        self.embed_dim = d_model

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Args:
            tgt (torch.tensor): Tensor of shape (bs, nchan, seq_len, embed_dim)
            memory (torch.tensor): Tensor of shape (bs, seq_len, embed_dim)
            tgt_mask (torch.tensor): Tensor of shape (bs, seq_len, seq_len)
            memory_mask (torch.tensor): Tensor of shape (bs, seq_len, seq_len)
            tgt_key_padding_mask (torch.tensor): Tensor of shape (bs, seq_len)
            memory_key_padding_mask (torch.tensor): Tensor of shape (bs, seq_len)
        """
        channel_proc_out = torch.stack([
            chan_layer(
                self.dropout(tgt[:, c, :, :]),
                self.dropout(tgt[:, c, :, :]),
                tgt_key_padding_mask,
                is_causal=tgt_mask is not None,
                attn_mask=tgt_mask,
            ) for c, chan_layer in enumerate(self.chan_processing_layers)
        ], dim=1)

        # channel_proc_out : (bs, nchan, seq_len, embed_dim)

        indices = list(range(self.num_channels))
        channel_inter_out = []
        for c, chan_layer in enumerate(self.chan_interaction_layers):
            src_channel = channel_proc_out[:, c, :, :]
            tgt_channels = channel_proc_out[:, indices[:c] + indices[c+1:], :, :]

            out = chan_layer(
                self.dropout(src_channel),
                self.dropout(tgt_channels),
                tgt_key_padding_mask,
                is_causal=tgt_mask is not None,
                attn_mask=tgt_mask,
            )
            channel_inter_out.append(out)

        channel_inter_out = torch.stack(channel_inter_out, dim=1)  # (bs, nchan, seq_len, d_model)

        channel_input_out = torch.stack([
            chan_layer(
                self.dropout(memory),
                self.dropout(channel_inter_out[:, c, :, :]),
                memory_key_padding_mask,
                is_causal=memory_mask is not None,
                attn_mask=memory_mask,
            ) for c, chan_layer in enumerate(self.chan_input_layers)
        ], dim=1)

        channel_input_out = self.layer_norm(channel_input_out)
        ff_out = self.feed_forward(self.dropout(channel_input_out))
        output = channel_input_out + ff_out

        return output


class ArtSpeechTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_articulators: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        num_feat: int = 100,
        dropout: float = 0.,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.src_embedding = nn.Embedding(vocab_size, embed_dim)

        self.tgt_embedding = nn.Sequential(
            nn.LayerNorm(num_feat),
            nn.Linear(num_feat, embed_dim),
            nn.ReLU()
        )

        self.pos_encoding = PositionalEncoding(
            embed_dim=embed_dim,
            dropout=dropout,
        )

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            transformer_encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=True,
        )

        transformer_decoder_layer = MultiChannelTransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            nchan=num_articulators,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            transformer_decoder_layer,
            num_layers=num_layers,
        )

        self.linear = nn.Sequential(
            nn.LayerNorm(num_articulators * embed_dim),
            nn.Linear(num_articulators * embed_dim, embed_dim),
            nn.ReLU(),
        )

        self.predictors = nn.ModuleList([
            ArticulatorPredictor(embed_dim, num_feat // 2) for _ in range(num_articulators)
        ])

        start = torch.zeros(1, 1, num_articulators, num_feat)
        self.register_buffer("start", start, persistent=False)  # (1, 1, num_art, 2 * num_samples)

    @property
    def total_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        src,
        tgt,
        src_attn_mask=None,
        tgt_attn_mask=None,
        memory_mask=None,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Args:
            src (torch.tensor): Tensor of shape (bs, seq_len)
            tgt (torch.tensor): Tensor of shape (bs, seq_len, num_channels, num_feat)
            src_lengths (List): List with the length of each sequence in the src batch
            tgt_lengths (List): List with the length of each sequence in the tgt batch
        """
        bs, seq_len, num_channels, num_feat = tgt.shape

        src_embed = self.src_embedding(src)
        src_pos_embed = self.pos_encoding(src_embed)

        src_attn_mask = torch.repeat_interleave(src_attn_mask, self.num_heads, dim=0)
        tgt_attn_mask = torch.repeat_interleave(tgt_attn_mask, self.num_heads, dim=0)

        encoder_out = self.encoder(
            src_pos_embed,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=False,
        )

        output = self._generate_one_step(
            tgt=tgt,
            memory=encoder_out,
            tgt_mask=tgt_attn_mask,
            memory_mask=src_attn_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return output

    def generate(
        self,
        src,
        src_key_padding_mask
    ):
        bs, seq_len = src.shape

        if self.start is None:
            raise Exception("Can not generate without a START token setup.")

        src_embed = src_embed = self.src_embedding(src)
        src_pos_embed = self.pos_encoding(src_embed)

        encoder_out = self.encoder(
            src_pos_embed,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=False,
        )

        tgt = torch.repeat_interleave(self.start, bs, dim=0)
        for _ in range(seq_len):
            next_tgt = self._generate_one_step(
                tgt=tgt,
                memory=encoder_out,
                memory_key_padding_mask=src_key_padding_mask
            )

            bs, curr_seq_len, num_channels, _, num_samples = next_tgt.shape
            next_tgt = next_tgt.reshape(bs, curr_seq_len, num_channels, 2 * num_samples)

            tgt = torch.cat([
                tgt,
                next_tgt[:, [-1], :, :]
            ], dim=1)

        bs, curr_seq_len, num_channels, num_features = tgt.shape
        tgt = tgt.reshape(bs, curr_seq_len, num_channels, 2, num_samples)

        return tgt


    def _generate_one_step(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        bs, seq_len, num_channels, num_feat = tgt.shape

        tgt_embed = self.tgt_embedding(tgt).permute(0, 2, 1, 3)
        tgt_pos_embed = self.pos_encoding(
            tgt_embed.reshape(
                bs * num_channels,
                seq_len,
                self.embed_dim
            )
        )
        tgt_pos_embed = tgt_pos_embed.reshape(
            bs,
            num_channels,
            seq_len,
            self.embed_dim,
        )

        decoder_out = self.decoder(
            tgt=tgt_pos_embed,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        decoder_out = decoder_out.permute(0, 2, 1, 3)  # (bs, seq_len, num_channels, num_feat)

        features = decoder_out.reshape(bs, seq_len, num_channels * self.embed_dim)
        features = self.linear(self.dropout(features))

        features = torch.stack([
            predictor(features) for predictor in self.predictors
        ], dim=2)
        next_tgt = torch.sigmoid(features)

        return next_tgt
