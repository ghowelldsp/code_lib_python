#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Signals

@author: G. Howell
"""

import numpy as np

def createToneSignals(
        amp:np.array, 
        f:np.array, 
        N:int, 
        nSignals:int=1, 
        fs:int=48000, 
        dtype=np.float32):
    """ Create Tone Signals

    Args:
        amp (np.array):             1D array of amplitudes for each channel. If less than the number of signals then
                                    the initial values will be used across all signals.
        f (np.array):               1D array of frequencies for each channel. If less than the number of signals then
                                    the initial values will be used across all signals.
        N (int):                    Number of samples.
        nSignals (int, optional):   Number of signals. Defaults to 1.
        fs (int, optional):         Sample rate [Hz]. Defaults to 48000.
        dtype (datatype, optional): Datatype. Defaults to np.float32.

    Returns:
        x (np.array):               2D array of signals. Takes the form [nChannels][nSamples]
        t (np.array):               1D array of time vector [seconds].
    """
    
    # if either the amplitude or frequency values are a single value, then create array of equal values for all signals
    if (len(amp) < nSignals):
        amp = np.full([nSignals], amp)
    else:
        amp = np.asarray(amp)
    if (len(f) < nSignals):
        f = np.full([nSignals], f)
    else:
        f = np.asarray(f)
    
    # create empty data vector
    x = np.zeros([nSignals, N], dtype=dtype);
    
    # time vector
    t = np.arange(0,N) * (1/fs)

    # create signals
    for i in range(nSignals):
        x[i,:] = amp[i] * np.sin(2*np.pi*f[i]*t)
        
    return x, t

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    nSignals = 1
    fs = 48000
    N = 256
    f = [1000, 2000]
    amp = [1.0, 1.0]
    
    x, t = createToneSignals(amp, f, N, nSignals, fs);
    
    # plot signals
    plt.plot(t, x.T)
    plt.grid() 
    plt.show()
    