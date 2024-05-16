#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

import numpy as np
import scipy.signal as sig

def isStable(a:np.array,
             disp:bool=False):
    """ Check Stability
    
    Check to see if as IIR filter is stable

    Args:
        a (np.array)
            A coefficients (denominator)
        disp (bool)
            display stability results message
    """
    
    stable = 1
    
    p = sig.tf2zpk(1, a)[1]
    
    if (np.max(np.abs(sig.tf2zpk(1, a)[1])) > 1):
        stable = 0
        
    if disp:
        if stable:
            print('The filter is stable')
        else:
            print('The filter is unstable')
        
    return stable

def isMinPhase(b:np.array,
               a:np.array,
               disp:bool=False):
    """ Check Minimum Phase
    
    Check to see if as IIR filter is minimum phase

    Args:
        b (np.array)
            B coefficient (numerator)
        a (np.array)
            A coefficients (denominator)
        disp (bool)
            display stability results message
    """
    
    minPhase = 1
    
    z, p, _ = sig.tf2zpk(b, a)
    
    if (np.max(np.abs(p)) > 1) and (np.max(np.abs(z)) > 1):
        minPhase = 0
        
    if disp:
        if minPhase:
            print('The filter is minimum phase')
        else:
            print('The filter is not mimumum phase')
        
    return minPhase

def checkFilter(b:np.array,
                a:np.array,
                disp:bool=False):
    """ Check Filter

    Checks the filter to see if it is stable and minimum phase

    Args:
        b (np.array)
            B coefficient (numerator)
        a (np.array)
            A coefficients (denominator)
        disp (bool)
            display results messages
    """
    
    stable = isStable(a, True)
    minPhase = isMinPhase(b, a, True)
    
    return stable, minPhase

if __name__ == "__main__":
    
    a = np.array([1, 4])
    b = np.array([5, 7])
    
    stable = isStable(a, True)
    minPhase = isMinPhase(b, a, True)
    
    a = np.array([6, 4])
    b = np.array([5, 3])
    
    checkFilter(b, a, True)
