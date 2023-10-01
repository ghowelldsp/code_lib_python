#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ghowell
"""

from scipy import signal as sig
import numpy as np

def fir_create(noTaps, fc, fs, N):
    """
    creates an fir filter
    
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