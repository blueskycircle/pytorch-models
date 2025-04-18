import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, example_x):
        super(LogisticRegression, self).__init__()
        input_dim = example_x.shape[-1]  # Deduce number of predictors
        # Initialize fixed effects vector β (including intercept)
        self.beta = nn.Parameter(torch.zeros(input_dim + 1, 1))

    def forward(self, x):
        # Append a column of ones for the intercept
        x_with_intercept = torch.cat([torch.ones(x.size(0), 1), x], dim=1)
        logits = x_with_intercept @ self.beta  # Compute the logit: Xβ
        return logits

    def negative_log_likelihood(self, x, y):
        logits = self.forward(x)
        # Ensure y has the same shape as logits
        y = y.view_as(logits)
        # Compute predictions using the sigmoid function
        p = torch.sigmoid(logits)
        # For numerical stability, add a small epsilon inside the logarithms
        eps = 1e-8
        loss = -(y * torch.log(p + eps) + (1 - y) * torch.log(1 - p + eps)).sum()
        return loss
