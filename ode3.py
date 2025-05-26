#### SDOF oscillator

import torchdiffeq
import torch
import numpy as np
import matplotlib.pyplot as plt

# Select device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SDOF(torch.nn.Module):
    def __init__(self, u, v, device=device):
        super(SDOF, self).__init__()
        self.device = device  # Store device info

        self.u = u.to(device)
        self.v = v.to(device)

        self.duration = 10.0

        self.time = torch.linspace(0, self.duration, 400, device=device)

    def forward(self, t, state):

        X, V = state.to(self.device)
     
        k = 64.0     # w0 = 8 
        omega_f = 5   # forcing frequency

        t = t.to(self.device)

        forcing = ( (self.u[0]**2) * torch.cos(omega_f * t) + (self.v[0]**3) * torch.sin(omega_f * t))

        ddX = (- k * X) + forcing 

        return torch.stack([V, ddX]).to(self.device)

