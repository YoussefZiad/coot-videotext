import os
import sys
import numpy as np
import h5py

directory = ''

h5 = h5py.File("video_feat_100m.h5", "w")

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    arr = np.load(f)

    h5[filename[:len(filename)-4]] = arr

h5.close()

