import os
import yaml
import argparse
from joblib import Parallel, delayed
from planar_flow import *
from train import *
from helpers import *
from lsf import *

class Scaler(nn.Module):
	""" Custom Linear layer but mimics a standard linear layer """
	def __init__(self, scale=1):
		super().__init__()
		log_scale = torch.Tensor(np.log(scale)*np.ones(1))
		self.log_scale = nn.Parameter(log_scale)

	def forward(self):
		return self.log_scale

def main(seed_value, config_data):
    n_batch = config_data['parameters']['batch_size']
    n_epoch = config_data['parameters']['epochs']
    n_flow = config_data['model']['nflow']
    rel_func = config_data['lsm']['func_name']
    ndim = int(config_data['lsm']['ndim'])
    lr = float(config_data['optimizer']['LR'])
    verbose = config_data['output']['verbose']
    save = config_data['output']['save']
    path_save = './saves/files/'
    logdet_weight = float(config_data['regularizer']['LogDet'])
    save_name = 'Seed_' + str(seed_value) + '_flow_' + str(n_flow) + '_batch_' + str(n_batch) + '_epochs_' + str(n_epoch) + '_logdet_' + str(logdet_weight) + '_ndim_' + str(ndim) + '_lr_' + str(lr)

    # load data
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    logger_path = path_save + save_name +'.txt'
    print_file = open(logger_path, 'w')

    # Device selection
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define priors
    prior = torch.distributions.MultivariateNormal(torch.zeros(ndim).to(device), torch.eye(ndim).to(device))

    # define model
    importance_sampler = PlanarFlow(ndim, K=n_flow).to(device)
    #scaler = Scaler(scale=np.sqrt(ndim)).to(device)

    # define optimizers
    from torch.optim import Adam, SGD
    optimizer = Adam(importance_sampler.parameters(), lr = lr, amsgrad=True, weight_decay=1e-5)
    #optimizer = SGD(importance_sampler.parameters(), lr=lr, weight_decay=1e-5, momentum=0.9)


    
    # optimizer = Adam(importance_sampler.parameters(), lr = lr, amsgrad=True)

    # define limit state function
    LSF = LimitStateFunction(func=rel_func)
    func = LSF.lsmf
    func2 = LSF.actual_lsmf

    # Variables to store evolution of losses
    model_train(n_epoch, n_batch, ndim, device, importance_sampler, prior, func, func2, logdet_weight,  
                optimizer, verbose, print_file, save, path_save, save_name)  #save insted of False

def do_train(seed_value, config_data):
    import torch
    import torch.optim as optim
    import numpy as np
    import random

    torch.set_default_dtype(torch.float64)

    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main(seed_value, config_data)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', help='Location of configuration data', type=str, required=True)
    args = vars(parser.parse_args())
    print(args)
    config_data=load_config(args['configs'])
    
    seeds = list(range(1))
    results = Parallel(n_jobs=1)(delayed(do_train)(i, config_data) for i in seeds)   #seeds instead of 1


    n_batch = config_data['parameters']['batch_size']
    n_epoch = config_data['parameters']['epochs']
    n_flow = config_data['model']['nflow']
    rel_func = config_data['lsm']['func_name']
    ndim = int(config_data['lsm']['ndim'])
    lr = float(config_data['optimizer']['LR'])
    verbose = config_data['output']['verbose']
    save = config_data['output']['save']
    path_save = './saves/files/'
    logdet_weight = float(config_data['regularizer']['LogDet'])
    
    pF = np.zeros((100, ), dtype=np.float64)
    pF1 = np.zeros((100, ), dtype=np.float64)
    var_PF = np.zeros((100, ), dtype=np.float64)



    for seed in range(100):
        print(seed)
        file_name = 'Seed_'+str(seed)+'_flow_' + str(n_flow) + '_batch_' + str(n_batch) + '_epochs_' + str(n_epoch) + '_logdet_' + str(logdet_weight) + '_ndim_' + str(ndim)+ '_lr_' + str(lr) + '.h5'
        with h5py.File(path_save + file_name, 'r') as f:
            pF[seed] = np.asarray(f['est_pf'], dtype=np.float64)
            pF1[seed] = np.asarray(f['est_pf1'], dtype=np.float64)
            var_PF[seed] = np.asarray(f['var'], dtype=np.float64)[-1]
        f.close()

    with h5py.File('./saves/postprocess_flow_' + str(n_flow)+ '.h5', 'w') as f:
        f.create_dataset('pf', data=pF)
        f.create_dataset('pf1', data=pF1)
        f.create_dataset('var', data=var_PF)
    f.close()

    print('Complete.')