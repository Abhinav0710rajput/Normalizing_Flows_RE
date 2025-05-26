import torchdiffeq
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Select device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define system parameters
m = 6e4  # mass (kg)
k = 5e6  # stiffness
f = 0.05  # damping ratio
T = 0.69  # natural period (s)
c = 2 * m * f * np.sqrt(k/m)  # damping coefficient
a = 0.1  # hysteresis parameter
uy = 0.04  # yielding displacement (m)
duration = 8.0  # duration for response calculation (s)

# Bouc-Wen model parameters
A = 1
n = 3  # power for Bouc-Wen
gamma = 1 / (2 * (uy**n))
beta = 1 / (2 * (uy**n))

# Ground acceleration parameters (white noise)
S = 0.005
n_freq = 300  # number of frequency points
dw = 30 * np.pi / n_freq  # frequency step

class HystereticOscillator(torch.nn.Module):
    def __init__(self, u_j, u_hat_j, device=device):
        super(HystereticOscillator, self).__init__()
        self.device = device  # Store device info

        # Move all parameters to the correct device
        self.m = torch.tensor(m, dtype=torch.float32, device=device)
        self.c = torch.tensor(c, dtype=torch.float32, device=device)
        self.k = torch.tensor(k, dtype=torch.float32, device=device)
        self.a = torch.tensor(a, dtype=torch.float32, device=device)
        self.A = torch.tensor(A, dtype=torch.float32, device=device)
        self.gamma = torch.tensor(gamma, dtype=torch.float32, device=device)
        self.beta = torch.tensor(beta, dtype=torch.float32, device=device)
        self.n = torch.tensor(n, dtype=torch.float32, device=device)

        # Move input tensors to the correct device
        self.u_j = u_j.to(device)
        self.u_hat_j = u_hat_j.to(device)

        self.n_freq = n_freq
        self.time = torch.linspace(0, duration, 400, device=device)

        # Convert S and dw to tensors
        self.S = torch.tensor(S, dtype=torch.float32, device=device)
        self.dw = torch.tensor(dw, dtype=torch.float32, device=device)

    def forward(self, t, state):
        X, V, Z = state.to(self.device)  # Ensure state is on correct device



        
        # Time should be on the correct device
        t = t.to(self.device)

        # Ground acceleration
        omega_j = torch.arange(1, (self.n_freq // 2) + 1, device=self.device) * self.dw  # Shape: (n_freq // 2 - 1,)
        sqrt_term = torch.sqrt(2 * self.S * self.dw)  # Scalar, already on device

        # Compute cosine and sine terms
        cos_term = torch.cos(omega_j * t)
        sin_term = torch.sin(omega_j * t)

        # Compute ground acceleration
        ground_acc = sqrt_term * (self.u_j * cos_term + self.u_hat_j * sin_term)
        ground_acc = torch.sum(ground_acc)  # Ensure it's a scalar

        # Dynamics of the oscillator
        acc = ((-ground_acc * self.m) - (self.c * V) - self.k * (self.a * X + (1 - self.a) * Z)) / self.m
        dZ = (-self.gamma * torch.abs(V) * (torch.abs(Z)**(self.n-1)) * Z -
              self.beta * (torch.abs(Z)**self.n) * V + self.A * V)
        

        # os.system('clear')

        # print(X, "XXX")
        # print(V, "VVV")
        
        # print(V.shape, "vshape")
        # print(acc.shape, "accshape")
        # print(dZ.shape, "dzshape")

        return torch.stack([V, acc, dZ]).to(self.device)

