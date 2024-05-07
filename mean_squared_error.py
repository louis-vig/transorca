import numpy as np
import pickle
import torch

def read_from_file(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

orca_predictions = read_from_file('orca_predictions.pickle')
trans_predictions = read_from_file('trans_predictions.pickle')
orca_experiments = read_from_file('orca_experiments.pickle')
trans_experiments = read_from_file('trans_experiments.pickle')

orca_predictions = np.array(orca_predictions)
trans_predictions = np.array(trans_predictions)
orca_experiments = np.array(orca_experiments)
trans_experiments = np.array(trans_experiments)

trans_loss = (
    (
        trans_predictions[~np.isnan(trans_experiments)]
        - trans_experiments[~np.isnan(trans_experiments)]
    )
    ** 2
).mean()

orca_loss = (
    (
        orca_predictions[~np.isnan(orca_experiments)]
        - orca_experiments[~np.isnan(orca_experiments)]
    )
    ** 2
).mean()

print(f"Transformer Loss = {trans_loss}")
print(f"ORCA Loss = {orca_loss}")

