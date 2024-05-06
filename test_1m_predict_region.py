import transorca_predict
transorca_predict.load_resources(models=["1M"], use_cuda=False)
from transorca_predict import *
outputs = process_region('chr8', 110404000, 111404000, hg38, window_radius=500000, custom_models=["h1esc_1m", "h1esc_trans_1m"], file='./chr17_110404000_111404000', use_cuda=False)
print(outputs)