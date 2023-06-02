import torch.nn as nn


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
