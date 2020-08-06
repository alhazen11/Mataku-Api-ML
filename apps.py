import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
import pickle
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v4 import *

plt.rcParams['figure.figsize'] = (4.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

parameters=load_obj("parameters")