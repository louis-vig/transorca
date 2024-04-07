import transorca_predict
transorca_predict.load_resources(models=['32M'])

from transorca_predict import *
outputs = process_region('chr9', 110404000, 111404000, hg38, window_radius=16000000, file='./chr17_110404000_111404000')

