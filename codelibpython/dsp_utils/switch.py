#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Switch

@author: G. Howell
"""

import numpy as np

class switch():
    """ Switch
    
    Enables switching between two sets of channel (A and B) ensuring a smooth transistion between the two sets by
    using a fade. Note, the switch operate independantly between the two sets, so channel 1 in set A fades to channel 2
    in set B, channel 2 in set A fades to channel 2 in set B, etc.
    """
    
    def __init__(self,
                 initPosition:str='A',
                 fadeTime:float=1.0,
                 fs:float=48000):
        """ Init
        
        Parameters
        ----------
        initPosition : str
            Initial switch position, either 'A' or 'B'
        fadeTime : float
            The fade time in seconds when switching between two the two channel sets.
        fs : float
            Sample rate [Hz].
        """
        
        # set initial gain
        if (initPosition == 'A'):
            self.gain = 1.0
        elif (initPosition == 'B'):
            self.gain = 0.0
        else:
            raise ValueError('invalid switch position')
        
        # set other values
        self.position = 'A'
        self.gainIncCurr = 0.0
        self.setFadeTime(fadeTime, fs)
        
    def setFadeTime(self,
                    fadeTime:float=1.0,
                    fs:float=48000):
        """ Set Fade Time
        
        Sets the fade time.

        Parameters
        ----------
        fadeTime : float
            Fade time in seconds. Defaults to 1.0.
        fs : float
            Sample rate. Defaults to 48000.
        """
        
        # calc gain increment value
        nFadeSamples = np.round(fadeTime * fs)
        self.gainInc = 1.0 / nFadeSamples
        
    def setPosition(self,
                    position:str):
        """ Set Position

        Parameters
        ----------
        position : str
            Position of switch, either 'A' or 'B'.
        """
        
        if (position != self.position):
            if (position == 'A'):
                self.gainIncCurr = self.gainInc
            elif (position == 'B'):
                self.gainIncCurr = -self.gainInc
            else:
                raise ValueError('invalid switch position')
            
        self.position = position
    
    def process(self,
                xA:np.array,
                xB:np.array):
        """ Process
        
        Parameters
        ----------
        xA : np.array [channels][samples]
            Channel set A input data.
        xB : np.array [channels][samples]
            Channel set B input data.
            
        Returns
        -------
        y : np.array [channels][samples]
            Output data.
        """
        
        assert xA.shape == xB.shape, 'Channel A and B signal are not of the same shape'
        
        _, nSamples = xA.shape
        
        # if the gain increment value is zero then simply pass through eihter the A of B channels set dependant on the
        # value of the gain
        if self.gainIncCurr == 0.0:
            if self.gain == 1.0:
                y = xA
            elif self.gain == 0.0:
                y = xB
        else:
            y = np.zeros(xA.shape)
            for i in range(nSamples):
            
                # gain increment values
                self.gain += self.gainIncCurr
                
                # check gain limits
                if (self.gain > 1.0):
                    self.gain = 1.0
                    self.gainIncCurr = 0.0
                elif (self.gain < 0.0):
                    self.gain = 0.0
                    self.gainIncCurr = 0.0
                
                # apply gain
                y[:,i] = self.gain * (xA[:,i] - xB[:,i]) + xB[:,i]
            
        return y
        
if __name__ == "__main__":
    
    import createSignals as cs
    import matplotlib.pyplot as plt
    
    # input signal parameters
    nSignals = 2
    ampA = np.linspace(0.2, 1.0, nSignals)
    fA = np.linspace(1000, 2000, nSignals)
    ampB = np.linspace(0.5, 0.5, nSignals)
    fB = np.linspace(1000, 2000, nSignals)
    blockLen = 32
    nBlocks = 8
    nSamples = blockLen * nBlocks
    fs = 48e3
    
    # create input signals
    xA, t = cs.createToneSignals(ampA, fA, nSamples, nSignals, fs)
    xB, t = cs.createToneSignals(ampB, fB, nSamples, nSignals, fs)
    
    # initialise switch
    fadeTime = blockLen / fs
    sw = switch('A', fadeTime, fs)
    
    # process data
    y = np.zeros(xA.shape)
    for i in range(nBlocks):
        
        if (i == 2):
            sw.setPosition('B')
        
        blockIdx = np.arange(blockLen * i, blockLen * (i + 1))
        y[:,blockIdx] = sw.process(xA[:,blockIdx], xB[:,blockIdx])
    
    # plotting
    fig, axs = plt.subplots(nSignals, 2, squeeze=False)
    fig.subplots_adjust(hspace=0.75)
    
    for i in range(nSignals):
        
        axs[i,0].plot(t, xA[i,:], label='A')
        axs[i,0].plot(t, xB[i,:], label='B')
        axs[i,0].set_title(f'Channel {i+1} - A & B Signals') 
        axs[i,0].grid()
        axs[i,0].set_xlim(t[0], t[-1])

        axs[i,1].plot(t, y[i,:], label='A')
        axs[i,1].set_title(f'Channel {i+1} - Output Signal') 
        axs[i,1].grid()
        axs[i,1].set_xlim(t[0], t[-1])
    
    axs[i,0].legend()
    plt.show()
