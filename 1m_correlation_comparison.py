import transorca_predict
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
USE_CUDA = False
WINDOW_RADIUS = 500_000
WINDOW_SIZE = 2 * WINDOW_RADIUS
CHR = 'chr9'
CHR9_SIZE = 138_394_717

transorca_predict.load_resources(models=["1M"], use_cuda=USE_CUDA)
from transorca_predict import *

def process_region_wrapped(start, end):
    return process_region(
        CHR,
        start,
        end,
        hg38,
        window_radius=WINDOW_RADIUS,
        custom_models=["h1esc_1m", "h1esc_trans_1m"],
        use_cuda=USE_CUDA,
        )

orca_predictions = []
trans_predictions = []
orca_experiments = []
trans_experiments = []

for start in range(0, CHR9_SIZE - WINDOW_RADIUS, WINDOW_RADIUS):
    print(f"running window {start}")
    end = start + WINDOW_SIZE
    output = process_region_wrapped(start, end)
    orca_predictions.append(output['predictions'][0])
    trans_predictions.append(output['predictions'][1])
    orca_experiments.append(output['experiments'][0])
    trans_experiments.append(output['experiments'][1])

orca_predictions = np.array(orca_predictions).reshape(-1)
trans_predictions = np.array(trans_predictions).reshape(-1)
orca_experiments = np.array(orca_experiments).reshape(-1)
trans_experiments = np.array(trans_experiments).reshape(-1)

corr = pearsonr(
        trans_predictions[~np.isnan(trans_experiments)],
        trans_experiments[~np.isnan(trans_experiments)],
    )[0]
print(f"CORR = {corr}")

indices = np.random.choice(trans_experiments[~np.isnan(trans_experiments)].shape[0], 10_000, replace=False)

plt.xlabel('Prediction')
plt.ylabel('Experiment')
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