class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        x_norm = (x - self.mean) / self.std
        return x_norm

    def inverse(self, x_norm):
        x = (x_norm * self.std) + self.mean
        return x
