#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decibel functions

@author: G. Howell
"""

import numpy as np

def linToDb(linVal:'float | np.array'):
    """ Linear To dB
    
    Converts linear values to decibels values
    
    Parameters
    ----------
    linVal : float
        Linear input value
    
    Returns
    -------
    dB value : float
        dB output
    """ 

    return 20*np.log10(abs(linVal))

def dbToLin(dbVal:'float | np.array'):
    """ dB To Linear
    
    Converts dB values to linear values
    
    Parameters
    ----------
    linVal : float
        Linear input value
    
    Returns
    -------
    dB value : float
        dB output
    """ 

    return 10**(dbVal/20)

if __name__ == "__main__":
    
    print(f'linToDb(0.5) = {linToDb(0.5)}')
    print(f'dbToLin(6) = {dbToLin(int(6))}')
