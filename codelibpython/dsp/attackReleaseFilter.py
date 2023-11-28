#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

import numpy as np
from scipy import signal as sig

def timeConstToCoeff(tau, fs):
    """
    Time Constant To Coefficient Calculation
    
    :param  tau:            Time constant [s]
    :param  fs:             Sample Rate [Hz]
    
    :return coefficient:    Calculated filter coefficient value
    """
    
    return np.exp(-1/(fs * tau))


def attackReleaseFilter(x, fs, attackTime, releaseTime):
    """
    Attack / Release Filters
    
    Creates a attack / release filter and processes the input signal. The attack / release filter is essentially an 
    lowpass IIR filter. As the time constant value is decreased the filter fc is increased, hence allowing more high
    frequencies to pass. In the time domain lower frequencies have a slower rate of change, meaning they are
    more easily reproduced by the filter. In short, the time constant has the effect of making the filter response 
    faster or slower to impulse like behaviour in time domain signals.
    
    Referenced from Zolzer, Udo., DAFX: Digital Audio Effects.
    
    :param  x:              Input signal in the form of a numpy array where dimension are [channels, samples]
    :param  fs:             Sample rate [Hz]
    :param  attackTime      Attack time constant [seconds]
    :param  releaseTime     Release time constant [seconds]
    
    :return y               Output signal
    """
    
    # get number of channels and shape
    nChannels, nSamples = x.shape
    
    # calculates the attack and release coefficients
    attackCoeff = timeConstToCoeff(attackTime, fs)
    releaseCoeff = timeConstToCoeff(releaseTime, fs)
    
    y = np.zeros([nChannels, nSamples])
    for i in range(nChannels):
        
        # reset previous output sample for each channel
        yPrevious = 0
        
        for j in range(nSamples):
            
            xCurrent = x[i,j]
            
            # select to use attack or release coefficient
            if yPrevious < xCurrent:
                coeff = attackCoeff
            else:
                coeff = releaseCoeff

            # filter signal
            yCurrent = (1-coeff)*xCurrent + coeff*yPrevious
            yPrevious = yCurrent
            
            y[i,j] = yCurrent
    
    return y


def attackReleasePlotResponse(attackTime, releaseTime, fs):
    """
    Plot Attack Release Filter
    
    :param  attackTime      Attack time constant [seconds]
    :param  releaseTime     Release time constant [seconds]
    :param  fs:             Sample rate [Hz]
    """
    
    # calculates the attack and release coefficients
    attackCoeff = timeConstToCoeff(attackTime, fs)
    releaseCoeff = timeConstToCoeff(releaseTime, fs)
    
    f, attackH = sig.freqz([1-attackCoeff, 0], [1, -attackCoeff], fs=fs)
    f, releaseH = sig.freqz([1-releaseCoeff, 0], [1, -releaseCoeff], fs=fs)
    
    fig, axs = plt.subplots(2, 1)
    
    axs[0].plot(f, 20*np.log10(attackH))
    axs[0].set_title('Attack Filter')
    
    axs[1].plot(f, 20*np.log10(releaseH))
    axs[1].set_title('Release Filter')
    
    for i in range(2):
        axs[i].grid()
        axs[i].set_xlim(f[0], f[-1])
    
    plt.show()
       
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    # input signal params
    f = 10
    fs = 48000
    N = 5000
    t = np.zeros([1,N])
    t[0,:] = np.arange(0,N) * (1/fs)
    
    # create input signal
    xTone = np.zeros([1,N])
    xTone[0,:] = np.sin(2*np.pi*f*t)
    
    # create step response
    xStep = np.zeros([1,N])
    xStep[0,100:100+1000] = 1
    
    # attack / release filtering
    attackTime = 200/fs
    releaseTime = 200/fs
    
    # processs data through attack release filter
    yTone = attackReleaseFilter(xTone, fs, attackTime, releaseTime)
    yStep = attackReleaseFilter(xStep, fs, attackTime, releaseTime)
    
    # plot frequency response
    attackReleasePlotResponse(attackTime, releaseTime, fs)
    
    # plotting
    fig, axs = plt.subplots(2, 1)
    
    axs[0].plot(t[0,:], xTone[0,:], label='input')
    axs[0].plot(t[0,:], yTone[0,:], label='output')
    axs[0].set_title('Tone')
    
    axs[1].plot(t[0,:], xStep[0,:], label='input')
    axs[1].plot(t[0,:], yStep[0,:], label='output')
    axs[1].set_title('Step')
    
    for i in range(2):
        axs[i].grid()
        axs[i].set_xlim(t[0,0], t[0,-1])
    
    plt.show()
