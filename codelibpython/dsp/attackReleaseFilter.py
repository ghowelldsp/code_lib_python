#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt

class attackReleaseFilter:
    """
    Attack / Release Filter
    
    Creates a attack / release filter and processes the input signal. The attack / release filter is essentially an 
    lowpass IIR filter. As the time constant value is decreased the filter fc is increased, hence allowing more high
    frequencies to pass. In the time domain lower frequencies have a slower rate of change, meaning they are
    more easily reproduced by the filter. In short, the time constant has the effect of making the filter response 
    faster or slower to impulse like behaviour in time domain signals.
    
    Referenced from Zolzer, Udo., DAFX: Digital Audio Effects.
    """
    
    def __init__(self, fs, attackTime, releaseTime, initialVal):
        """
        :param  fs:             Sample rate [Hz]
        :param  attackTime      Attack time constant [seconds]
        :param  releaseTime     Release time constant [seconds]
        :param  initialVal      Initial gain value [value between 0.0 -> 1.0]
        """
        self.fs = fs
        self.attackTime = attackTime
        self.releaseTime = releaseTime
        self.attackCoeff = self.timeConstToCoeff(attackTime)
        self.releaseCoeff = self.timeConstToCoeff(releaseTime)
        self.initialVal = initialVal
        
    def timeConstToCoeff(self, tau):
        """
        Time Constant To Coefficient Calculation
        
        :param  tau:            Time constant [s]
        :param  fs:             Sample Rate [Hz]
        
        :return coefficient:    Calculated filter coefficient value
        """
        
        return np.exp(-1/(self.fs * tau))
    
    def process(self, x):
        """
        Process a signal through the attack / release filter
        
        :param  x:              Input signal in the form of a numpy array where dimension are [channels, samples]
        
        :return y               Output signal
        """
        
        # get number of channels and shape
        nChannels, nSamples = x.shape
        
        y = np.zeros([nChannels, nSamples])
        for i in range(nChannels):
            
            # reset previous output sample for each channel
            yPrevious = self.initialVal
            
            for j in range(nSamples):
                
                xCurrent = x[i,j]
                
                # select to use attack or release coefficient
                if yPrevious < xCurrent:
                    coeff = self.attackCoeff
                else:
                    coeff = self.releaseCoeff

                # filter signal
                yCurrent = (1-coeff)*xCurrent + coeff*yPrevious
                yPrevious = yCurrent
                
                y[i,j] = yCurrent
        
        return y
    
    def plotResponse(self, x, y, title):
        """
        Plots Input and Output Signals
        """
        
        # get number of channels and shape
        nChannels, nSamples = x.shape
        
        # time vector
        t = np.zeros([1,nSamples])
        t[0,:] = np.arange(0,nSamples) * (1/self.fs)
        
        # plotting
        fig, axs = plt.subplots(nChannels, 1, squeeze=False)
        
        for i in range(nChannels):
            axs[i,0].plot(t[0,:], x[i,:], label='input')
            axs[i,0].set_title(title)
            axs[i,0].plot(t[0,:], y[i,:], label='output')
            axs[i,0].grid()
            axs[i,0].set_xlim(t[0,0], t[0,-1])
        
        plt.show()

    def plotFilters(self):
        """
        Plot Filter Response
        """
        
        # calculate frequency response
        f, attackH = sig.freqz([1-self.attackCoeff, 0], [1, -self.attackCoeff], fs=fs)
        f, releaseH = sig.freqz([1-self.releaseCoeff, 0], [1, -self.releaseCoeff], fs=fs)
        
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
    initialVal = 0.0
    
    # initialise ar filter
    arFlt = attackReleaseFilter(fs, attackTime, releaseTime, initialVal)
    
    # processs data through ar filter
    yTone = arFlt.process(xTone)
    yStep = arFlt.process(xStep)
    
    # plot time domain response
    arFlt.plotResponse(xTone, yTone, 'Tone')
    arFlt.plotResponse(xStep, yStep, 'Step')
    
    arFlt.plotFilters()
