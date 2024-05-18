#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

import numpy as np
import scipy.signal as sig

def parametricEq(fc:float,
                 Q:float,
                 gainDb:float,
                 fs:int):
    """ Parametric EQ
    
    Parametric EQ based upon work by RBJ

    Parameters
    ==========
        fc : float
            Cutoff frequency [Hz]
        Q : float
            Quality factor
        gainDb : float
            Gain in decibels.
        fs : int
            Samplerate [Hz]

    Returns
    =======
        bZ : float
            B coefficients
        aZ : float
            A coefficients
    """

    wc = 2*np.pi*fc
    A = 10**(gainDb/40)

    bS = np.array([1, A/Q*wc, wc**2])
    aS = np.array([1, 1/(A*Q)*wc, wc**2])
    
    bZ, aZ = sig.bilinear(bS, aS, fs)
    
    return bZ, aZ