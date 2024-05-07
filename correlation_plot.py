import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import pearsonr

np.random.seed(42)

def read_from_file(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

orca_predictions = read_from_file('orca_predictions.pickle')
trans_predictions = read_from_file('trans_predictions.pickle')
orca_experiments = read_from_file('orca_experiments.pickle')
trans_experiments = read_from_file('trans_experiments.pickle')

orca_predictions = np.array(orca_predictions).reshape(-1)
trans_predictions = np.array(trans_predictions).reshape(-1)
orca_experiments = np.array(orca_experiments).reshape(-1)
trans_experiments = np.array(trans_experiments).reshape(-1)

trans_corr = pearsonr(
        trans_predictions[~np.isnan(trans_experiments)],
        trans_experiments[~np.isnan(trans_experiments)],
    )[0]
print(f"Transformer CORR = {trans_corr}")
orca_corr = pearsonr(
        orca_predictions[~np.isnan(orca_experiments)],
        orca_experiments[~np.isnan(orca_experiments)],
    )[0]
print(f"ORCA CORR = {orca_corr}")

indices = np.random.choice(trans_experiments[~np.isnan(trans_experiments)].shape[0], 10_000, replace=False)

plt.xlabel('Prediction')
plt.ylabel('Experiment')
plt.title('TransORCA')
plt.xticks([-2, -1, 0, 1, 2])
plt.yticks([-2, -1, 0, 1, 2])
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.plot([-2, 2], [-2, 2], 'r--')
plt.plot(
    trans_predictions[indices],
    trans_experiments[indices],
    'k.',
    markersize=2,
)
plt.savefig('prediction_vs_experiment.pdf')
plt.close()

plt.xlabel('Prediction')
plt.ylabel('Experiment')
plt.title('ORCA')
plt.xticks([-2, -1, 0, 1, 2])
plt.yticks([-2, -1, 0, 1, 2])
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.plot([-2, 2], [-2, 2], 'r--')
plt.plot(
    orca_predictions[indices],
    orca_experiments[indices],
    'k.',
    markersize=2,
)
plt.savefig('prediction_vs_experiment_orca.pdf')