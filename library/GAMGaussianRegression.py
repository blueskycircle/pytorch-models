import torch
import torch.nn as nn
import math

# A simple cubic polynomial basis expansion for a single predictor.
def cubic_basis(x):
    # x has shape [n, 1]
    # Returns a design matrix of shape [n, 4] containing [1, x, x^2, x^3]
    return torch.cat([torch.ones_like(x), x, x**2, x**3], dim=1)

class GAMGaussianRegression(nn.Module):
    def __init__(self):
        super(GAMGaussianRegression, self).__init__()
        # We use a cubic basis expansion so there are 4 basis functions
        self.num_basis = 4
        # Learnable coefficients for the basis functions (including intercept)
        self.beta = nn.Parameter(torch.randn(self.num_basis, 1))
        # Log-transformed standard deviation for numerical stability
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x):
        # Transform the predictor with the cubic basis
        B = cubic_basis(x)   # B has shape [n, 4]
        # Compute predictions: μ = B @ β
        mu = B @ self.beta
        return mu
    
    def sigma(self):
        # Compute σ from its log-scale value
        return torch.exp(self.log_sigma)
    
    def negative_log_likelihood(self, x, y):
        # Compute predictions and sigma
        mu = self.forward(x)
        sigma = self.sigma()
        # Negative log likelihood for a Gaussian:
        # NLL = Σ [ log(σ) + 0.5*((y-μ)/σ)^2 ] + constant
        nll = (torch.log(sigma) + 0.5 * ((y - mu) / sigma)**2).sum() \
            + 0.5 * math.log(2 * math.pi) * y.numel()
        return nll
