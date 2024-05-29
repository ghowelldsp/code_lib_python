#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

import matplotlib.pyplot as plt
import numpy as np
from pymatreader import read_mat
import os

def checkImpedance(impedFileName:str, 
                   saveData:bool=False):
    
    # get the file extension
    fileName, fileExt = os.path.splitext(impedFileName)
    
    # load the data from the file
    if (fileExt == '.mat'):
        data = read_mat(impedFileName)['impedData']
    
    # get data
    freq = np.asarray(data['freq'], dtype=np.float32)
    mag = data['mag']
    phase = data['phase']
    
    assert ((len(freq) == len(mag)) or (len(freq) == len(phase))), 'freq data is not the same length as the magnitude \
                                                                     and / or phase'
    
    # create complex impedance vector
    cpxImped = mag * np.exp(1j * (phase * np.pi/180.0))
    
    # save data
    if saveData:
        np.savez(fileName, f=freq, Z=cpxImped)
    
    # plot data
    plt.subplot(2,1,1)
    plt.plot(freq, mag)
    plt.grid()
    plt.title('Impedance Magnitude')
    plt.ylabel('Mag [Ohm]')
    plt.xlim(freq[0], freq[-1])
    
    plt.subplot(2,1,2)
    plt.plot(freq, phase)
    plt.grid()
    plt.title('Impedance Phase')
    plt.ylabel('Phase [deg]')
    plt.xlabel('freq [Hz]')
    plt.xlim(freq[0], freq[-1])
    
    plt.show()
    
    return data

if __name__ == "__main__":
    
    impedFileName = "impedTestData/testData.mat"
    
    checkImpedance(impedFileName, saveData=True)
