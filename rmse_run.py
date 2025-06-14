import os
import h5py
import re
import numpy as np
from collections import defaultdict
import pandas as pd



# Define paths

def create_rmse():
    source_folder = "saves/files"
    output_folder = "rmse_files"

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Regex to extract seed and hyperparameters
    pattern = re.compile(r"Seed_(\d+)_(.*)\.h5")

    # Group files by hyperparameters (excluding Seed value)
    file_groups = defaultdict(list)

    # Collect all files and sort them into groups
    for filename in os.listdir(source_folder):
        match = pattern.match(filename)
        if match:
            seed, hyperparams = match.groups()
            file_groups[hyperparams].append((int(seed), filename))

    # Process each hyperparameter group
    for hyperparams, files in file_groups.items():
        files.sort()  # Ensure files are in order by Seed

        est_pf_values = []
        est_pf1_values = []

        for seed, filename in files:
            filepath = os.path.join(source_folder, filename)
            with h5py.File(filepath, 'r') as f:
                est_pf_values.append(f["est_pf"][()])
                est_pf1_values.append(f["est_pf1"][()])

        # Convert to numpy arrays
        est_pf_values = np.array(est_pf_values)
        est_pf1_values = np.array(est_pf1_values)

        # Save new file
        output_filename = os.path.join(output_folder, f"{hyperparams}.h5")
        with h5py.File(output_filename, 'w') as f:
            f.create_dataset("est_pf", data=est_pf_values)
            f.create_dataset("est_pf1", data=est_pf1_values)

        print(f"Saved {output_filename}")

###############################################
###############################################

def calc_rmse():

    rmse_folder = "rmse_files"

    # Define regex to extract hyperparameters from the filename
    pattern = re.compile(
        r"flow_(\d+)_batch_(\d+)_epochs_(\d+)_logdet_([\d\.]+)_ndim_(\d+)_lr_([\d\.]+)\.h5"
    )

    columns = ["flow", "batch", "epochs", "logdet", "ndim", "lr", "est_pf", "est_pf1", "true_prob"]
    df = pd.DataFrame(columns=columns)


    # Loop through each .h5 file in rmse_files
    for filename in os.listdir(rmse_folder):
        match = pattern.match(filename)
        if match:
            # Extract hyperparameters as separate variables
            flow = int(match.group(1))
            batch = int(match.group(2))
            epochs = int(match.group(3))
            logdet = float(match.group(4))
            ndim = int(match.group(5))
            lr = float(match.group(6))

            filepath = os.path.join(rmse_folder, filename)

            # Read the .h5 file
            with h5py.File(filepath, 'r') as f:
                est_pf = np.array(f["est_pf"])
                est_pf1 = np.array(f["est_pf1"])

            if(ndim==300):
                true_prob = 0.83e-3
            if(ndim==2):
                true_prob = 0.83e-3 ######

            df = pd.concat(
            [df, pd.DataFrame([[flow, batch, epochs, logdet, ndim, lr, est_pf, est_pf1, true_prob]], columns=columns)],
            ignore_index=True
            )

            print(f"Processing: {filename}")
            print(f"Flow: {flow}, Batch: {batch}, Epochs: {epochs}, Logdet: {logdet}, Ndim: {ndim}, LR: {lr}")
            print(f"est_pf shape: {est_pf.shape}, est_pf1 shape: {est_pf1.shape}")

            

    df['rb'] = df.apply(lambda row: (np.mean(row['est_pf1']) - row['true_prob']) / row['true_prob'], axis=1)
    df['nRMSE'] = df.apply(lambda row: np.sqrt(
    (row['rb'] ** 2) +  
    ((np.mean(row['est_pf1']) ** 2) / (row['true_prob'] ** 2)) *  
    (np.var(row['est_pf1']) / (np.mean(row['est_pf1']) ** 2))
    ), axis=1)

    # df = df.drop(['est_pf1', 'est_pf', 'rb', 'true_prob'], axis=1)
    
    df.to_csv("nRMSE.csv")

if __name__ == "__main__":
    create_rmse()
    calc_rmse()
    print("RMSE calculation completed and saved to nRMSE.csv")







    
