import torch
import torch.nn as nn


class PoissonModel(nn.Module):
    def __init__(self, example_x):
        super(PoissonModel, self).__init__()
        input_dim = example_x.shape[-1]  # Deduce number of predictors
        # Initialize fixed effects vector β (including intercept)
        self.beta = nn.Parameter(torch.randn(input_dim + 1, 1))

    def forward(self, x):
        # Append a column of ones for the intercept
        x_with_intercept = torch.cat([torch.ones(x.size(0), 1), x], dim=1)
        eta = x_with_intercept @ self.beta
        # Use the log link to compute the rate λ = exp(η)
        return torch.exp(eta)

    def negative_log_likelihood(self, x, y):
        # Compute the negative log likelihood for Poisson regression (omitting the constant log(y!))
        x_with_intercept = torch.cat([torch.ones(x.size(0), 1), x], dim=1)
        eta = x_with_intercept @ self.beta
        lambda_pred = torch.exp(eta)
        # Poisson negative log-likelihood: ∑ [λ - y * η]
        return (lambda_pred - y * eta).sum()
