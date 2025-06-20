# Normalizing Flows for Reliability Analysis

_______________________

This projects contains the implementation of Normalizing Flows for estimating probabilities of failure in structural reliability analysis. The primary goal is to use a Planar Flow model as an importance sampling distribution to efficiently estimate very small failure probabilities.

## File Structure

- `main.py`: The main script to run the training and evaluation.
- `train.py`: Contains the training loop for the Normalizing Flow model.
- `planar_flow.py`: Implementation of the Planar Flow model.
- `lsf.py`: Defines the Limit State Function used for reliability analysis.
- `helpers.py`: Contains helper functions for loading data and configurations.
- `config.yaml`: Configuration file for hyperparameters and model settings.
- `requirements.txt`: A list of required Python packages.
- `saves/`: Directory where trained models and results are saved.
- `logs_new/`: Directory for logging experiment outputs.
- `postprocess.ipynb`, `Rmse_analysis.ipynb`: Jupyter notebooks for analyzing the results.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Abhinav0710rajput/Normalizing_Flows_RE.git
    cd Normalizing_Flows_RE
    ```

2.  It is recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The `config.yaml` file is used to set the hyperparameters and other settings for the model and training.

```yaml
lsm:
  func_name: linear
  ndim: 300 
model:
  affine: true
  nflow: 15
optimizer:
  LR: 0.001
output:
  save: true
  verbose: true
parameters:
  batch_size: 300
  epochs: 30
regularizer:
  LogDet: 1.0
```

- `lsm.func_name`: The name of the limit state function to use.
- `lsm.ndim`: The number of dimensions of the problem.
- `model.nflow`: The number of layers in the Planar Flow model.
- `optimizer.LR`: The learning rate for the Adam optimizer.
- `output.save`: Whether to save the trained model and results.
- `output.verbose`: Whether to print progress during training.
- `parameters.batch_size`: The batch size for training.
- `parameters.epochs`: The number of epochs for training.
- `regularizer.LogDet`: The weight of the log-determinant of the Jacobian in the loss function.

## Usage

To run the training, execute the `main.py` script with the path to the configuration file:

```bash
python main.py --configs config.yaml
```

The script will train the Planar Flow model using the settings from the `config.yaml` file. The trained models and results will be saved in the `saves/files/` directory. The filenames of the saved files are structured to include the hyperparameters used for that run.

After training, the script will post-process the results from multiple runs (with different random seeds) and save the aggregated results in the `saves/` directory.

## Author

Abhinav Rajput


