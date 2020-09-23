import numpy as np

# dataset.py
TRAIN_RATIO = 0.8

# train.py
IMAGE_SIZE = 224
start_lr = 0.001

# inference.py
NUM_CLASSES = 10

# adjust prediction
adjust_ratio = np.asarray([0.181,0.209,0.093,0.068,0.113,0.063,0.116,0.048,0.058,0.050])
