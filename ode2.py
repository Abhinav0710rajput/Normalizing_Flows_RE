import torchdiffeq
import torch
import numpy as np
import matplotlib.pyplot as plt

# Select device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class hyperbolic(torch.nn.Module):
    def __init__(self, u, v, device=device):
        super(hyperbolic, self).__init__()
        self.device = device  # Store device info

        self.u = u.to(device)
        self.v = v.to(device)

        self.duration = 10.0

        self.time = torch.linspace(0, self.duration, 400, device=device)

        self.term = (u**2 * v**2) - 2


    def forward(self, t, state):
        X, V = state.to(self.device)  # Ensure state is on correct device

        # Time should be on the correct device
        t = t.to(self.device)

        f = (2 + self.term[0] / 50.0 + (3 * self.term[0]) / 100.0 * t**2)

        ddX = (f - t * V - X)


        return torch.stack([V, ddX]).to(self.device)

