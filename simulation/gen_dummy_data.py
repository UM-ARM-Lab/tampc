"""Generate dummy data to test model training and preprocessing"""
from meta_contact import cfg
import os
import numpy as np
import scipy.io

filename = os.path.join(cfg.DATA_DIR, "pushing", "dummy.mat")
N = 1000
nx = 5
nu = 2
ny = 3
# 0 centered gaussian noise
sigma = 0.1

# give some order to X
X = np.random.rand(N, nx + nu)
X[:, 0] *= 2
X[:, 1] = (X[:, 1] - 0.5) * 0.5
X[:, 2] = (X[:, 3] - 0.25) * 3
X = np.sort(X, axis=0)
# generate some random linear system
params = np.random.rand(nx + nu, ny)
Y = (X @ params) + np.random.randn(N, ny) * sigma
d = {'X': X[:, 0:nx], 'U': X[:, nx:nx + nu], 'Y': Y, 'contact': np.zeros((N, 1))}
scipy.io.savemat(filename, mdict=d)
