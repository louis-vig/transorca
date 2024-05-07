import transorca_predict
import numpy as np
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
        file=f"./chr9_{start}_{start+WINDOW_SIZE}",
        model_labels=["ORCA", "TransORCA"]
        )

np.random.seed(42)
starts = np.random.choice(CHR9_SIZE-WINDOW_SIZE, 3, replace=False)

for start in starts:
    print(f"running window {start}", flush=True)
    end = start + WINDOW_SIZE
    output = process_region_wrapped(start, end)