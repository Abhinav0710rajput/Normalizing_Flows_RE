import numpy as np
import torch
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
import h5py
from ode import *
from lsf import *
import os
import matplotlib.pyplot as plt
import re

from datetime import datetime

def model_train(n_epoch, n_batch, ndim, device, importance_sampler, prior, func, func2, logdet_weight_target, optimizer, verbose, print_file, save, path_save, save_name):

    loss_list = []
    loss_prior_list = []
    loss_data_list = []
    logdet_list = []
    loss2_list = []
    pf_list = []
    pf1_list = []
    scaling_val = []
    
    beta2_start = 20
    beta1_start = 1

    alpha_start = 1
    alpha_end = 10
	
    T_cool = 0.1*n_epoch
    scheduler = MultiStepLR(optimizer, milestones=[T_cool], gamma=1.0) #intially 1.0  d d



    #pretraining steps:
    for _ in range(10):
        z_sample = torch.randn(n_batch, ndim).to(device=device)
        theta_sample, logdet = importance_sampler(z_sample)
        loss = -torch.mean(logdet)

        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(importance_sampler.parameters(), 0.01)
        optimizer.step()
        


    for k in range(n_epoch):

        if k+1 <= n_epoch - T_cool:
            alpha = alpha_start + (alpha_end - alpha_start)*((k+1) / (n_epoch-T_cool))
            beta1 = beta1_start + (1.0 - beta1_start)*((k+1) / (n_epoch-T_cool))
            beta2 = beta2_start - (beta2_start - logdet_weight_target)*((k+1) / (n_epoch-T_cool))
        else:
            alpha = alpha_end
            beta1 = 1.0   ###############  what are these??
            beta2 = logdet_weight_target  ############# what are these ??
        
        z_sample = torch.randn(n_batch, ndim).to(device=device)    ########### initially sampled z 
        
        ########################

        theta_sample, logdet1 = importance_sampler(z_sample)

        #############

        if(k == 0):
            sample_ = torch.randn(1000, ndim).to(device=device)
            all_samples, _ = importance_sampler(sample_)
            #print(all_samples.shape, " <- collected samples")
        else:
            sample_ = torch.randn(1000, ndim).to(device=device)
            transformed_sample, _ = importance_sampler(sample_)
            all_samples = torch.cat([all_samples, transformed_sample], dim=0)
            #print(all_samples.shape, " <- collected samples")

        torch.save(all_samples, 'all_samples.pt')

        #############

        logdet = torch.squeeze(logdet1)
        
        loss_data = torch.log(func(theta_sample, alpha))
        loss_prior = prior.log_prob(theta_sample)
            
        loss = -1*torch.mean(loss_data) - beta1*torch.mean(loss_prior) - beta2*torch.mean(logdet)
        loss2 = torch.var(loss_data + loss_prior + logdet - prior.log_prob(z_sample))

        #print(loss2)

        pf = torch.mean((func2(theta_sample).squeeze().to(device))*torch.exp((loss_prior + logdet - prior.log_prob(z_sample))))
        pf1 = torch.mean(torch.exp(loss_data + loss_prior + logdet - prior.log_prob(z_sample)))

        optimizer.zero_grad()
        loss.backward()

        #torch.nn.utils.clip_grad_norm_(importance_sampler.parameters(), 0.01)
        optimizer.step()
        scheduler.step()

        loss_list.append(loss.detach().cpu().numpy())
        loss_data_list.append(torch.mean(loss_data).detach().cpu().numpy())
        loss_prior_list.append(torch.mean(loss_prior).detach().cpu().numpy())
        logdet_list.append(-torch.mean(logdet).detach().cpu().numpy())
        loss2_list.append(loss2.detach().cpu().numpy())
        pf_list.append(pf.detach().cpu().numpy())
        pf1_list.append(pf1.detach().cpu().numpy())


        log_message = f"epoch: {k:}, loss: {loss_list[-1]:.2f}, loss2: {loss2_list[-1]:.2f}, pf:{pf_list[-1]:1.2e}, pf1:{pf1_list[-1]:1.2e}, loss data: {loss_data_list[-1]:.2f}, loss prior: {loss_prior_list[-1]:.2f}, logdet: {logdet_list[-1]:.2f}"
        print(log_message)

        pf = "training_log.csv"

        if not os.path.exists(pf):
            with open(pf, 'w') as f:
                f.write("epoch, loss, loss2, pf, pf1, loss_data, loss_prior, logdet\n")

        # Append the latest values to the file
        with open(pf, 'a') as f:
            f.write(f"{k}, {loss_list[-1]:.2f}, {loss2_list[-1]:.2f}, {pf_list[-1]:1.2e}, {pf1_list[-1]:1.2e}, "
                    f"{loss_data_list[-1]:.2f}, {loss_prior_list[-1]:.2f}, {logdet_list[-1]:.2f}\n")



    if save:  ## save is false for now
        # 1) move sampler to CPU
        importance_sampler_cpu = importance_sampler.to('cpu')

        # loop over seeds

        with torch.no_grad():
            for index_seed in range(10):     ###### 100 seeds initially
                # update seed in save_name 

                save_name_cpu = re.sub(r"Seed_\d+_", f"Seed_{index_seed}_", save_name)

                print(f"Calculating pf for {save_name_cpu} on CPU…")

                n_samples = 10000 #10000 initially

                # 2) sample z on CPU
                z_sample_cpu = torch.randn(n_samples, ndim, device='cpu')

                print("sanity check ", 1)

                # 3) run your sampler on CPU
                theta_sample_cpu, logdet1_cpu = importance_sampler_cpu(z_sample_cpu)

                print("sanity check ", 2)

                # squeeze logdet
                logdet_cpu = torch.squeeze(logdet1_cpu)

                print("sanity check ", 3)

                # everything from here stays on CPU
                loss_data_cpu   = torch.log(func(theta_sample_cpu, alpha_end))

                print("sanity check ", 4)

                device = torch.device("cpu")

                prior = torch.distributions.MultivariateNormal(torch.zeros(ndim).to(device), torch.eye(ndim).to(device))

                loss_prior_cpu  = prior.log_prob(theta_sample_cpu)

                print("sanity check ", 5)

                pf_cpu  = torch.mean(func2(theta_sample_cpu).squeeze() *
                                    torch.exp(loss_prior_cpu + logdet_cpu
                                            - prior.log_prob(z_sample_cpu)))
                pf1_cpu = torch.mean(torch.exp(
                            loss_data_cpu + loss_prior_cpu + logdet_cpu
                            - prior.log_prob(z_sample_cpu)
                        ))

                print(f"  pf  = {pf_cpu:1.2e}")
                print(f"  pf1 = {pf1_cpu:1.2e}")

                # 4) save everything (will be on CPU tensors, so .cpu() is a no-op)
                with h5py.File(path_save + save_name_cpu + '.h5', 'w') as f:

                    f.create_dataset('total', data=np.array(loss_list))
                    f.create_dataset('data', data=np.array(loss_data_list))
                    f.create_dataset('logdet', data=np.array(logdet_list))
                    f.create_dataset('prior', data=np.array(loss_prior_list))
                    f.create_dataset('var', data=np.array(loss2_list))
                    f.create_dataset('pf', data=np.array(pf_list))
                    f.create_dataset('pf1', data=np.array(pf1_list))
                    f.create_dataset('est_pf', data=pf_cpu.detach().cpu().numpy())
                    f.create_dataset('est_pf1', data=pf1_cpu.detach().cpu().numpy())
                    f.create_dataset('samples', data=theta_sample_cpu.detach().cpu().numpy())

        # 5) move sampler back to GPU so further training still works
        importance_sampler.to(device)

        
if __name__=='__main__':
    print('This function stores all the training scripts.')
