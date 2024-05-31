#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

import numpy as np

def timeToCoeffConst(time:float,
                     fs:int):
    """ Time To Coefficient Constant

    Parameters
    ----------
        time : float
            Time [seconds]
        fs : int
            Sample rate [Hz]
            
    Returns
    -------
        coeff : float
            Coefficient value
        oneMinusCoeff : float
            One minus coefficient value (1 - coeff)
    """

    fc = 1 / (2 * np.pi * time)

    coeff = np.exp(-2 * np.pi * fc/fs)
    oneMinusCoeff = 1 - coeff

    return coeff, oneMinusCoeff

if __name__ == "__main__":
    
    time = 0.1
    fs = 48000
    
    coeff, oneMinusCoeff = timeToCoeffConst(time, fs)
    
