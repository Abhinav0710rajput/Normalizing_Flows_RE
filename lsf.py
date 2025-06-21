import numpy as np
import torch
from ode import *
from ode2 import *
from ode3 import *
import torchdiffeq
import torch
import torchdiffeq
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing


torch.cuda.empty_cache()




def circular(theta):
    """
    Two dimensional limit state function with infinite failure modes.
    x_1 ~ N(0,1)
    x_2 ~ N(0,1)

    Cite: 
    1. Arief, Mansur, et al. "Certifiable Deep Importance Sampling for Rare-Event 
    Simulation of Black-Box Systems." arXiv preprint arXiv:2111.02204 (2021).

    Parameters
    ----------
    theta : torch.Tensor
        Tensor of shape (batch_size, n_dims)

    Returns
    -------
    torch.Tensor
        Tensor of shape (batch_size, )
    """
    return 5.60 - torch.sqrt(theta[:,0]**2 + theta[:,1]**2)


def linear(theta, threshold = 0.04):
    return 1. - theta[:, 0]/threshold

def linear2(theta, threshold = 9):
    return 1. - theta[:, 0]/threshold

def linear3(theta, threshold = 0.5):
    return 1. - theta[:, 0]/threshold


class LimitStateFunction:
    def __init__(self, func=None):
        self.func = func
        if self.func == 'circular':
            self.lsm = circular
            self.pf = 1.54975e-07
        elif self.func == 'linear':   # Hystrectic
            self.lsm = linear
        elif self.func == 'linear2':  # Quadratic
            self.lsm = linear2
        elif self.func == 'linear3':  # SDOF
            self.lsm = linear3
        else:      
            raise ValueError('Invalid limit state function.')
        
    def ode_solve(self, theta):
        device = theta.device  # Get the device of theta
        n_freq = 300  # theta.size(1)
        u_j = theta[:, :n_freq//2].to(device)  # Move to the correct device
        u_hat_j = theta[:, n_freq//2:].to(device)  # Move to the correct device
        initial_state = torch.tensor([0.0, 0.0, 0.0], device=device)  # Ensure initial_state is on the same device

        for sim in range(theta.size(0)):  # Can be parallelized
            if sim % 1 == 0:
                print("sim ", sim + 1, "  device", device)  

            oscillator = HystereticOscillator(u_j[sim, :], u_hat_j[sim, :], device = device)  # Move model to device

            

            oscillator = oscillator.to(device)


            solution = torchdiffeq.odeint(oscillator, initial_state, oscillator.time.to(device), method='rk4')
            final_displacement = solution[-1:, 0:1]

            if sim == 0:
                ans = final_displacement
            else:
                ans = torch.cat((ans, final_displacement), dim=0)

        return ans
    

    def ode_solve2(self, theta, ode=hyperbolic): # theta here should be the random variables
        device = theta.device  # Get the device of theta
        n_freq = 2  # theta.size(1)
        u = theta[:, :n_freq//2].to(device)  # Move to the correct device
        v = theta[:, n_freq//2:].to(device)  # Move to the correct device


        if(ode == SDOF):
            initial_state = torch.tensor([0.0, 0.0], device=device)
        else:
            initial_state = torch.tensor([2.0, 0.0], device=device)  # Ensure initial_state is on the same device
        
        for sim in range(theta.size(0)):  # Can be parallelized
            if sim % 1 == 0:
                print("sim ", sim + 1)

            oscillator = ode(u[sim, :], v[sim, :]).to(device)  # Move model to device

            solution = torchdiffeq.odeint(oscillator, initial_state, oscillator.time.to(device), method='rk4')
            final_displacement = solution[-1:, 0:1]

            if sim == 0:
                ans = final_displacement
            else:
                ans = torch.cat((ans, final_displacement), dim=0)

        return ans


    def lsmf(self, theta, alpha=10):
        theta_ = torch.zeros_like(theta)
        if self.lsm == linear:
            theta_ = self.ode_solve(theta)
        elif self.lsm == linear2:
            theta_  = self.ode_solve2(theta)
        elif self.lsm == linear3:
            theta_  = self.ode_solve2(theta, ode=SDOF)
        else:
            theta_ = theta
        performance = self.lsm(theta_)
        val = torch.sigmoid(-1*alpha*performance)
        return torch.squeeze(torch.clamp(val, min=10**-100, max=1.0))
        
    def actual_lsmf(self, theta):
        if self.lsm == linear:
            theta = self.ode_solve(theta)
        elif self.lsm == linear2:
            theta  = self.ode_solve2(theta)
        elif self.lsm == linear3:
            theta  = self.ode_solve2(theta, ode=SDOF)
        performance = self.lsm(theta)
        mask = torch.zeros(performance.shape)
        mask[performance <= 0] = 1
        return mask


if __name__=='main':
    a = torch.rand(300, 2) 
    l = LimitStateFunction(func = 'linear')
    b = l.lsmf(a)
    for i in range(100):
        print(b[i])
    print('lsf.py is not supposed to be run as main')
