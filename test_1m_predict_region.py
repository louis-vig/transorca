import transorca_predict
transorca_predict.load_resources(models=["1M"])
from transorca_predict import *
outputs = process_region('chr9', 110404000, 111404000, hg38, window_radius=500000, file='./chr17_110404000_111404000')