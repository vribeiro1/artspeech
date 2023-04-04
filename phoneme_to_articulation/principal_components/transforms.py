import torch
import torch.nn as nn


class Encode(nn.Module):
    def __init__(self, encoder_cls, state_dict_filepath, device, **kwargs):
        super().__init__()

        self.encoder = encoder_cls(**kwargs)
        encoder_state_dict = torch.load(state_dict_filepath, map_location=device)
        self.encoder.load_state_dict(encoder_state_dict)
        self.encoder.to(device)

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        return self.encoder(x)


class Decode(nn.Module):
    def __init__(self, decoder_cls, state_dict_filepath, device, **kwargs):
        super().__init__()

        self.decoder = decoder_cls(**kwargs)
        decoder_state_dict = torch.load(state_dict_filepath, map_location=device)
        self.decoder.load_state_dict(decoder_state_dict)
        self.decoder.to(device)

        for parameter in self.decoder.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        return self.decoder(x)


class InputTransform(nn.Module):
    def __init__(self, transform, device, activation=None, **kwargs):
        super().__init__()
        self.transform = transform
        self.transform.to(device)
        self.activation = activation

        for parameter in self.transform.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        output = self.transform(x)
        if self.activation is not None:
            output = self.activation(output)
        return output
