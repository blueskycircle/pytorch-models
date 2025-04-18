import torch
import torch.nn as nn
import numpy as np


class GaussianModel(nn.Module):
    def __init__(self, example_x):
        super(GaussianModel, self).__init__()
        input_dim = example_x.shape[-1]  # Deduce the number of predictors
        self.beta = nn.Parameter(
            torch.randn(input_dim + 1, 1)
        )  # β vector includes intercept
        self.log_sigma = nn.Parameter(
            torch.tensor(0.0)
        )  # log(σ) for numerical stability

    def forward(self, x):
        x_with_intercept = torch.cat([torch.ones(x.size(0), 1), x], dim=1)
        mu = x_with_intercept @ self.beta  # μ = Xβ
        return mu

    def sigma(self):
        return torch.exp(self.log_sigma)  # Convert log(σ) to σ

    def negative_log_likelihood(self, x, y):
        mu = self(x)
        sigma = self.sigma()
        # Negative log-likelihood for normal distribution
        return (torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2).sum() + 0.5 * np.log(
            2 * np.pi
        ) * y.numel()
