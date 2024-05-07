import transorca_predict
import pickle
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
    print(f"running window {start}", flush=True)
    end = start + WINDOW_SIZE
    output = process_region_wrapped(start, end)
    orca_predictions.append(output['predictions'][0])
    trans_predictions.append(output['predictions'][1])
    orca_experiments.append(output['experiments'][0])
    trans_experiments.append(output['experiments'][1])

def dump_to_file(obj, name):
    with open(f"{name}.pickle", 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

dump_to_file(orca_predictions, "orca_predictions")
dump_to_file(trans_predictions, "trans_predictions")
dump_to_file(orca_experiments, "orca_experiments")
dump_to_file(trans_experiments, "trans_experiments")

print('successfully dumped to files')