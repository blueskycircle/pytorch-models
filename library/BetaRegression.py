import torch
import torch.nn as nn


class BetaRegression(nn.Module):
    def __init__(self, example_x):
        super(BetaRegression, self).__init__()
        input_dim = example_x.shape[-1]
        self.beta = nn.Parameter(torch.randn(input_dim + 1, 1))  # Mean parameters
        self.log_phi = nn.Parameter(torch.tensor(1.0))  # Precision parameter

    def forward(self, x):
        x_with_intercept = torch.cat([torch.ones(x.size(0), 1), x], dim=1)
        logits = x_with_intercept @ self.beta
        mu = torch.sigmoid(logits)  # Ensures 0 < μ < 1
        return mu

    def phi(self):
        return torch.exp(self.log_phi)  # Ensures precision φ > 0

    def negative_log_likelihood(self, x, y):
        mu = self(x)
        phi = self.phi()

        # Calculate shape parameters
        p = mu * phi
        q = (1 - mu) * phi

        # Beta negative log-likelihood
        nll = -torch.sum(
            torch.lgamma(phi)
            - torch.lgamma(p)
            - torch.lgamma(q)
            + (p - 1) * torch.log(y)
            + (q - 1) * torch.log(1 - y)
        )

        return nll
