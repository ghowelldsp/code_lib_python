#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

import numpy as np
import matplotlib.pyplot as plt

def lagrange_interpolation(x, y, xInterp):
    """Lagrange Interpolation

    Args:
        x (np.array): Input data, 1D array.
        y (np.array): Output data, 1D array.
    """
    
    # check x and y are same length
    if (len(x) != len(y)):
        raise ValueError("Arrays must have the same length")
    
    # number of samples in and original and interpolated arrays
    nSamples = len(x)
    nInterpSamples = len(xInterp)
    
    # reshape x values to 2d array
    x2d = np.reshape(x, [-1,1])
    
    # create a matrix of xi - xj values, where i is row and j is columns
    denomMat = x2d - x2d.T
    
    # calculate the product of the rows ignoring 0 values
    denomProd = np.prod(np.ma.masked_equal(denomMat,0), axis=1).compressed()
    
    # calculate the ceofficients
    ceoffs = y / denomProd
    
    # calculate each new interpolation sample
    yInterp = np.zeros((nInterpSamples))
    for i in range(nInterpSamples):
        
        # create array of current interpolation sample
        xItp = np.full((nSamples,1), xInterp[i])
        
        # calculate a matrix of the form
        # [ {(x-x0), (x-x1), ... , (x-xn)}, 
        #   {(x-x0), (x-x1), ... , (x-xn)}, 
        #   ... ]
        diff = xItp - x2d.T

        # fill the diagonal with unity
        np.fill_diagonal(diff, 1)
        
        # calculate the product of the rows and sum the column
        yInterp[i] = np.sum(np.prod(diff, axis=1) * ceoffs) 
    
    return yInterp
    
if __name__ == "__main__":
    
    # create a signal
    N = 10
    x = np.arange(0,N)
    xIterp = np.arange(0,N)
    y = np.sin(np.pi * np.linspace(0,1,N))
    
    yInterp = lagrange_interpolation(x, y, xIterp)
    
    # plotting
    plt.plot(x, y, label='original')
    plt.plot(xIterp, yInterp, label='interp')
    plt.grid()
    plt.legend()
    
    plt.show()
    
    