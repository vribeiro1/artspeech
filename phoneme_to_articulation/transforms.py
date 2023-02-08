class Normalize:
    def __init__(self, mean, std):
        """
        Args:
        mean (torch.tensor): Tensor of shape (N, M)
        std (torch.tensor): Tensor of shape (N, M)
        """
        self.mean = mean
        self.std = std

    def __call__(self, x):
        """
        Normalizes a tensor. Substracts the mean and divide by the standard deviation.

        Args:
        x (torch.tensor): Tensor of shape (*, N, M)
        """
        mean = self.mean.clone().to(x.device)
        std = self.std.clone().to(x.device)
        x_norm = (x - mean) / std
        return x_norm

    def inverse(self, x_norm):
        """
        De-normalizes a tensor. Multiplies by the standard deviation and add the mean.

        Args:
        x_norm (torch.tensor): Tensor of shape (*, N, M)
        """
        mean = self.mean.clone().to(x_norm.device)
        std = self.std.clone().to(x_norm.device)
        x = (x_norm * std) + mean
        return x
