import yaml
import itertools
import os
import logging

import random

from rmse_run import *

with open("config.yaml", "r") as f:
    base_config = yaml.safe_load(f)

param_grid = {
    "model.nflow": [40, 60],
    "parameters.batch_size": [20,50,100,200],
    "parameters.epochs": [50],
    "optimizer.LR": [0.001],
    "regularizer.LogDet": [1.5]
}

# Generate all combinations of hyperparameters
keys = list(param_grid.keys())
param_combinations = list(itertools.product(*param_grid.values()))

# Ensure log directory exists
os.makedirs("logs_new", exist_ok=True)

# Track the previous config file
previous_config_path = None

# Function to run experiment
def run_experiment(i, param_values):
    global previous_config_path

    print("Running config ", i)

    # Delete the previous config file (if it exists)
    if previous_config_path and os.path.exists(previous_config_path):
        os.remove(previous_config_path)

    # Create a new config dictionary
    config = base_config.copy()

    # Assign new hyperparameters
    for key, value in zip(keys, param_values):
        section, param = key.split(".")
        config[section][param] = value

    # Save modified config
    config_path = f"config__{i}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Update the previous config path
    previous_config_path = config_path

    # Create a unique log file name
    log_file = f"logs_new/exp_{i}_nflow{config['model']['nflow']}_batch{config['parameters']['batch_size']}_epochs{config['parameters']['epochs']}_lr{config['optimizer']['LR']}_logdet{config['regularizer']['LogDet']}.txt"

    # Set up logging
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.handlers = []  # Remove previous handlers
    file_handler = logging.FileHandler(log_file)
    logger.addHandler(file_handler)

    # Write hyperparameters to log file
    logger.info("### Experiment Configuration ###")
    logger.info(yaml.dump(config, default_flow_style=False))
    logger.info("\n### Training Log ###\n")

    # Run training script and log output
    os.system(f"python main.py --configs {config_path} >> {log_file} 2>&1")

random.shuffle(param_combinations)

# Run experiments sequentially
for i, param_values in enumerate(param_combinations):
    run_experiment(i, param_values)

os.remove(previous_config_path)

create_rmse()
calc_rmse()




