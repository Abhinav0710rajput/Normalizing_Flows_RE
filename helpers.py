import numpy as np
import scipy.io
import torch
import torch.nn as nn
import yaml

def load_config(config_file):
    with open(config_file,'r') as f:
        config_data=yaml.safe_load(f)
    return(config_data)

if __name__ == '__main__':
    print('This file contains all helper scripts for ptychography')    