import numpy as np
import torch

from tqdm import tqdm

TRAIN = "train"
VALID = "validation"
TEST = "test"


def run_autoencoder_epoch(phase, epoch, model, dataloader, optimizer, criterion, scheduler=None, fn_metrics=None, writer=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if fn_metrics is None:
        fn_metrics={}
    training = phase == TRAIN

    if training:
        model.train()
    else:
        model.eval()

    losses = []
    metrics_values = {metric_name: [] for metric_name in fn_metrics}
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - {phase}")
    for _, inputs, sample_weigths, _ in progress_bar:
        inputs = inputs.to(device)
        sample_weigths = sample_weigths.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs, latents = model(inputs)

            loss = criterion(
                outputs,
                latents,
                inputs,
                sample_weigths
            )

            if training:
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

            for metric_name, fn_metric in fn_metrics.items():
                metric_val = fn_metric(outputs, inputs)
                metrics_values[metric_name].append(metric_val.item())

        losses.append(loss.item())
        progress_bar.set_postfix(loss=np.mean(losses))

    mean_loss = np.mean(losses)
    loss_tag = f"{phase}/loss"
    if writer is not None:
        writer.add_scalar(loss_tag, mean_loss, epoch)

    info = {
        "loss": mean_loss
    }

    return info
