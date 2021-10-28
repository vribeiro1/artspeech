import torch

from vt_tracker.metrics import p2cp_mean


def p2cp_distance(outputs, targets):
    # outputs: torch.Size([bs, seq_len, N_art, 2, N_samples])
    # targets: torch.Size([bs, seq_len, N_art, 2, N_samples])
    bs, seq_len, N_art, _, N_samples = outputs.shape

    results = torch.zeros(0, seq_len, N_art)
    for batch_out, batch_target in zip(outputs, targets):
        batch_results = torch.zeros(0, N_art)
        for seq_out, seq_target in zip(batch_out, batch_target):
            seq_results = []
            for output, target in zip(seq_out, seq_target):
                output_transpose = output.transpose(1, 0)
                target_transpose = target.transpose(1, 0)

                p2cp = p2cp_mean(output_transpose.numpy(), target_transpose.numpy())
                seq_results.append(p2cp)

            batch_results = torch.cat([batch_results, torch.tensor(seq_results).unsqueeze(dim=0)])
        results = torch.cat([results, batch_results.unsqueeze(dim=0)])

    return results
