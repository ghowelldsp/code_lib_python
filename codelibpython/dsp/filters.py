#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ghowell
"""

from scipy import signal as sig
import numpy as np

def fir_create(noTaps, fc, fs, N):
    """
    Creates an fir filter
    
    :param noTaps: number of taps in the fir fitler
    :param fc: cutoff frequency [Hz]
    :param fs: sample rate [Hz]
    :param N: length of resultant frequency response
    
    :return b: fir coefficients
    :return H: complex full (pos & neg) frequency response
    :return f: frequency vector
    """ 

    # create filter coefficients
    b = sig.firwin(noTaps, fc/fs)

    # transform to frequency domain
    W, H = sig.freqz(b, 1, worN=N, whole=True)

    # calculates the frequency vector in Hz
    f = np.arange(0,N)*(fs/N)

    return b, H, f

def fir_fixed_point(b, x, q):
    """
    Performs fixed point fir filtering

    :param b: coefficients
    :param x: input signal
    :param q: number fractional point bits

    :return y: output signal
    """
    
    N = len(x)
    y = np.zeros([N,1])
    delayBuff = np.zeros([noTaps,1])

    for i in range(N):

        # circular shift delay buffer
        delayBuff = np.roll(delayBuff, 1)

        # add input sample to buffer start
        delayBuff[0] = x[i]

        # calculate output
        y[i] = np.dot(delayBuff.T, b) * 2**-q

    return y

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    import maths_basic as mb
    import dsp_fixedpoint as fp
    import utils_dataio as dio

    """
    Fir Filter
    """

    # filter settings 
    noTaps = 12
    fc = 500
    fs = 2000
    N = 2000

    # create fir filter
    b, H, f = fir_create(noTaps, fc, fs, N)

    # plot filter
    plt.plot(f, mb.db(H))
    plt.title('Filter Response')
    plt.xlabel('frequency [Hz]')
    plt.ylabel('magnitude [dB]')
    plt.grid()
    plt.xlim(0,fs/2)

    plt.show()

    """
    Fractional Point Filter
    """

    # signal settings
    fc = 100
    fs = 2000
    N = 100

    # create signal
    x, t = dio.createToneSig(1.0, fc, N, fs)
    
    # filter settings 
    noTaps = 12

    # create fir filter
    b, H, f = fir_create(noTaps, fc, fs, N)

    # scale to fixed point values
    fracLen = 15
    wordLen = 32
    bFp = fp.floatToFixed(b, fracLen, wordLen, True)
    xFp = fp.floatToFixed(x, fracLen, wordLen, True)

    # filter signal
    y = fir_fixed_point(bFp[0], xFp[0], fracLen)

    # scale output to float
    y = fp.fixedToFloat(y, fracLen)

    # plot output
    plt.plot(t, x)
    plt.plot(t, y)
    plt.title('Input vs Output Signals')
    plt.xlabel('time [s]')
    plt.ylabel('amplitude')
    plt.grid()

    plt.show()

    plt.plot(f, mb.db(H))
    plt.title('Filter Response')
    plt.xlabel('frequency [Hz]')
    plt.ylabel('magnitude [dB]')
    plt.grid()
    plt.xlim(0,fs/2)

    plt.show()
