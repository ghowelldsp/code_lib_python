#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

import numpy as np
import scipy.linalg as sla

def corrVector(x:np.array,
               y:np.array=None):
    """ Correlation Vector
    
    Calculates the correlation vector of the form,
    
        r_xy(k) = E[x(n) y^*(n-k)],    where k = 0, 1, 2, ...
        
    If only x(n) is used then the process is autocorrelation, r_xx, else if both x(n) and y(n) data are present, then
    cross-correlation, r_xy,  will be performed.

    Parameters
    ----------
    x : np.array
        1D array of input data.
    y : np.array
        1D array of input data. Defaults to None.
        
    Returns:
    r_xy : np.array
        1D array of correlation data.
    """
    
    # if no second vector is input perform auto-correlation
    if y is None:
        y = x
        
    assert len(x) == len(y), 'The vectors are not of equal length'
    
    # create correlation vector (positive indexs only)
    rx = np.correlate(x, y, mode='full')[len(x)-1:]
    
    return rx

def corrMatrix(x:np.array,
               y:np.array=None):
    """ Correlation Matrix
    
    Calculates the correlation matrix of the form,
    
                  | r(0)   r^*(-1) r^*(-2) | 
        R_xy(k) = | r(-1)  r(0)    r^*(-1) |
                  | r(-2)  r(-1)   r(0)    |
        
    If only x(n) is used then the process is autocorrelation, r_xx, else if both x(n) and y(n) data are present, then
    cross-correlation, r_xy,  will be performed.

    Parameters
    ----------
    x : np.array
        1D array of input data.
    y : np.array
        1D array of input data. Defaults to None.
        
    Returns:
    r_xy : np.array
        2D array of correlation matrix values.
    """
    
    # create correlation vector (positive indexs only)
    rx = corrVector(x, y)
    
    # create the correlation matrix
    Rx = sla.toeplitz(rx)
    
    return Rx

if __name__ == "__main__":
    
    # array
    x = np.array([1+1j, 2+1j, 3-1j])
    y = np.array([1+3j, 6+1j, 7-3j])
    
    # auto-correlation
    Rxx = corrMatrix(x)

    # cross-correlation
    Rxy = corrMatrix(x, y)
    
    # print output
    print(Rxx)
    print(Rxy)
