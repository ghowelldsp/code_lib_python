#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 13:49:48 2023

@author: ghowell
"""

import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, '../src')
import data_io as io

# create a signal
fs = 10000
[x,t] = io.createToneSig(1.0, 100, 1024, fs)

# write a wav file
io.writeWavFile(x, 'tone.wav', 1, 16, fs)

# reads a wav file
y = io.readWavFiles('tone.wav')

# error
print("data error = %.4f" % (np.sum(y-x)))

# plotting
plt.plot(t,x, label='input')
plt.plot(t,y, label='output')
plt.grid()
plt.legend(loc='upper right')

plt.show()