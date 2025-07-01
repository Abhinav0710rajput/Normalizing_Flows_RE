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
import shutil
from planar_flow import *
import argparse
from helpers import *



def save_ckp(state, checkpoint_dir, model_name):
    f_path = checkpoint_dir + model_name
    torch.save(state, f_path)

def load_ckp(checkpoint_fpath, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, scheduler, checkpoint['epoch']


def rep_train(n_batch, ndim, device, prior, func, func2, logdet_weight_target, config_data):

    n_epoch = 100

    n_flows = config_data['model']['nflow']
    lr = float(config_data['optimizer']['LR'])

    model_name = f"batch_{n_batch}_ndim_{ndim}_nflows_{n_flows}_lr_{lr}_logdet_{logdet_weight_target}.pt"
    checkpoint_dir = 're_train/'
    checkpoint_path = checkpoint_dir + model_name


    importance_sampler = PlanarFlow(ndim, K=n_flows).to(device)
    from torch.optim import Adam, SGD
    optimizer = Adam(importance_sampler.parameters(), lr = lr, amsgrad=True, weight_decay=1e-5)
    T_cool = 0.1*n_epoch #Total epochs = say 100
    scheduler = MultiStepLR(optimizer, milestones=[T_cool], gamma=1.0) #intially 1.0  d d

    epoch = 0
    s = False

    if os.path.exists(checkpoint_path):
        model, optimizer, scheduler, epoch = load_ckp(checkpoint_path, importance_sampler, optimizer, scheduler)
        print(f"Resuming training from epoch {epoch}")
        model.train() ########## IDK LOL
        s = True
        if epoch == n_epoch:
            print("Training completed")
            exit()
    else:
        epoch = 0
        print("Starting training from scratch")

    

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



    #pretraining steps:

    if not s:
        for _ in range(10):
            z_sample = torch.randn(n_batch, ndim).to(device=device)
            theta_sample, logdet = importance_sampler(z_sample)
            loss = -torch.mean(logdet)

            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(importance_sampler.parameters(), 0.01)
            optimizer.step()
        


    k = epoch

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

    #####################################################################################

    checkpoint = {
    'epoch': epoch + 1,
    'state_dict': importance_sampler.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict()
    }

    save_ckp(checkpoint, checkpoint_dir, model_name)

    loss_list.append(loss.detach().cpu().numpy())
    loss_data_list.append(torch.mean(loss_data).detach().cpu().numpy())
    loss_prior_list.append(torch.mean(loss_prior).detach().cpu().numpy())
    logdet_list.append(-torch.mean(logdet).detach().cpu().numpy())
    loss2_list.append(loss2.detach().cpu().numpy())
    pf_list.append(pf.detach().cpu().numpy())
    pf1_list.append(pf1.detach().cpu().numpy())


    log_message = f"epoch: {k:}, loss: {loss_list[-1]:.2f}, loss2: {loss2_list[-1]:.2f}, pf:{pf_list[-1]:1.2e}, pf1:{pf1_list[-1]:1.2e}, loss data: {loss_data_list[-1]:.2f}, loss prior: {loss_prior_list[-1]:.2f}, logdet: {logdet_list[-1]:.2f}"
    print(log_message)

    pf = f"re_train/batch_{n_batch}_ndim_{ndim}_nflows_{n_flows}_lr_{lr}_logdet_{logdet_weight_target}.csv"

    if not os.path.exists(pf):
        with open(pf, 'w') as f:
            f.write("epoch, loss, loss2, pf, pf1, loss_data, loss_prior, logdet\n")

    # Append the latest values to the file
    with open(pf, 'a') as f:
        f.write(f"{k}, {loss_list[-1]:.2f}, {loss2_list[-1]:.2f}, {pf_list[-1]:1.2e}, {pf1_list[-1]:1.2e}, "
                f"{loss_data_list[-1]:.2f}, {loss_prior_list[-1]:.2f}, {logdet_list[-1]:.2f}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', help='Location of configuration data', type=str, required=True)
    args = vars(parser.parse_args())
    print(args)
    config_data=load_config(args['configs'])
    n_batch = config_data['parameters']['batch_size']
    n_epoch = config_data['parameters']['epochs']
    n_flow = config_data['model']['nflow']
    rel_func = config_data['lsm']['func_name']
    ndim = int(config_data['lsm']['ndim'])
    lr = float(config_data['optimizer']['LR'])
    verbose = config_data['output']['verbose']
    save = config_data['output']['save']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior = torch.distributions.MultivariateNormal(torch.zeros(ndim).to(device), torch.eye(ndim).to(device))
    LSF = LimitStateFunction(func=rel_func)
    func = LSF.lsmf
    func2 = LSF.actual_lsmf
    logdet_weight = float(config_data['regularizer']['LogDet'])

    rep_train(n_batch, ndim, device, prior, func, func2, logdet_weight,config_data)









