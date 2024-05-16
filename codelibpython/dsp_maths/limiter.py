#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

import numpy as np
import matplotlib.pyplot as plt

# import attackReleaseFilter as arf

class limiter():
    """
    Limiter
    
    Performs independant limiting of multichannel data. The input signal is processed by a attack release peak
    detector, then a smoothing filter is applied to the determined gain decrease being applied to the input signal. 
    
    Referenced from Zolzer, Udo., DAFX: Digital Audio Effects.
    """
    def __init__(self, fs, threshold, attackTime, releaseTime, delayTime, nChannels, dtype=np.float32):
        """
        :param  fs:             sample rate [Hz]
        :param  threshold:      threshold value [dB]
        :param  attackTime:     attack time [seconds]
        :param  releaseTime:    release time [seconds]
        :param  delayTime:      delay time [seconds]
        """
        self.fs = fs
        self.attackTime = attackTime
        self.releaseTime = releaseTime
        self.delaySamples = int(delayTime * fs)
        self.nChannels = nChannels
        self.dtype = dtype
        
        # create delay buffer
        self.buffer = np.zeros([self.nChannels, self.delaySamples], dtype=dtype)
        self.xpeak = np.zeros([self.nChannels], dtype=dtype)
        self.g = np.ones([self.nChannels], dtype=dtype)
        
        # calculate linear gain
        self.threshold = np.float32(self.calculateGain(threshold))
        
        # TODO - fix
        # # initialise peak attack release filter (NOTE: the reversal of the ar coefficient due to changes in equality)
        # self.peakFlt = arf.attackReleaseFilter(fs, releaseTime, attackTime, 0.0)
        
        # # initialise attack release filter
        # self.smoothFlt = arf.attackReleaseFilter(fs, attackTime, releaseTime, 1.0)
        
    def calculateGain(self, threshold):
        """
        Calculate Threshold
        
        Converts the threshold from a dB to linear value
        
        :param  threshold:  limiter threshold [dB]
        
        :return linear threshold value
        """
        
        return 10**(threshold/20)
    
    # def process(self, x):
        """
        Process
        
        Old process function that uses attack release filter. Does not work properly atm.
        TODO - fix this
        """
    
        # # peak filter
        # peakVal = self.peakFlt.process(abs(x))
        
        # # peak comparison
        # peakMin = self.threshold/peakVal
        # peakMin[peakMin > 1] = 1
        
        # # smoothing filter
        # smoothVal = self.smoothFlt.process(peakMin)
        
        # # delay signal
        # nChannels, nSamples = x.shape
        # delayBuff = np.zeros([nChannels, nSamples])
        # delayBuff[:,self.delay:] = x[:,0:-self.delay]
        
        # # calculate output
        # y = smoothVal * delayBuff
        
    def process(self, x):
        """
        Process
        
        Processes input data through the limiter
        
        :param  x:  input data in the form of a numpy array with dimension [channels][samples]
        
        :return y:  output data in the same form as the input data  
        """
        
        self.x = x
        self.y = np.zeros(x.shape, dtype=self.dtype)
        nChannels, nSamples = x.shape
        
        for j in range(nChannels):
            for i in range(nSamples):
                
                # peak detector
                a = abs(x[j,i]).astype(self.dtype)
                if (a > self.xpeak[j]):
                    coeff = self.attackTime
                else:
                    coeff = self.releaseTime
                self.xpeak[j] = (1-coeff) * self.xpeak[j] + coeff * a
                
                # smoothing filter
                ratio = np.float32(self.threshold/self.xpeak[j])
                if (ratio > 1):
                    f = 1
                else:
                    f = ratio
                self.g[j] = (1-coeff) * self.g[j] + coeff * f
                
                # apply gain
                self.y[j,i] = self.g[j] * self.buffer[j,-1]
                
                # delay buffer
                self.buffer[j,:] = np.append(x[j,i], self.buffer[j,0:-1])
        
        return self.y
    
    def plot(self):
        """
        Plot
        
        Plots the input vs output data for each channel
        
        :param  x:  input data in the form of a numpy array with dimension [channels][samples]
        :param  y:  output data in the same form as the input data
        :param  title:  title of the plot
        """
        
        nChannels, nSamples = self.x.shape
        
        # time vector
        t = np.zeros([1,nSamples])
        t[0,:] = np.arange(0,nSamples) * (1/self.fs)
        
        # plotting
        fig, axs = plt.subplots(nChannels, 1)
        fig.subplots_adjust(hspace=0.75)
        
        for i in range(nChannels):
            axs[i].plot(t[0,:], self.x[i,:], label='input')
            axs[i].plot(t[0,:], self.y[i,:], label='output')
            axs[i].set_title(f'Limiter - Ch {i}')
            axs[i].grid()
            axs[i].set_xlim(t[0,0], t[0,-1])
        
        axs[i].legend()
        axs[i].set_xlabel('time [s]')
        
        plt.show()
        
if __name__ == "__main__":
    
    nChannels = 2
    
    # input signal params
    f = 100
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
    
    # limiter parameters
    threshold = -6
    attackTime = 0.3
    releaseTime = 0.01
    delayTime = 5/fs
    
    # initialise limiter
    limiterH = limiter(fs, threshold, attackTime, releaseTime, delayTime, nChannels)
    
    # combine input signals into one matrix
    x = np.zeros([nChannels, N])
    x[0,:] = xTone
    x[1,:] = xStep
    
    # processs data through ar filter
    y = limiterH.process(x)
    
    # plot time domain response
    limiterH.plot()
    