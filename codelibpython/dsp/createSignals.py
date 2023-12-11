#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Signals

@author: G. Howell
"""

import numpy as np
import matplotlib.pyplot as plt

class createSignals():
    """
    Create Signals
    
    Creates various signal including tones, noise, etc.
    """
    def __init__(self, amp, N, nChannels):
        """
        :param amp:         Amplitudes of the signals, in list form with one amplitude value for each channel
        :param N:           Number of samples for each signal
        :param fs:          Sample rate [Hz]
        :param nChannels    Number of channels 
        """
        self.amp = amp
        self.N = N
        self.nChannels = nChannels
    
    def createToneSignals(self, f, fs):
        """
        Creates tone signals
        
        :param f:       Tone frequencies, in list form with one frequency for each channel
        
        :return x:      Tone signals for each channel, in the form of a numpy matrix where [channels][samples]
        :return t:      Time vector [seconds]
        """
        
        self.f = f
        
        self.x = np.zeros([self.nChannels, self.N], dtype=np.float32);
        
        # time vector
        self.t = np.arange(0, self.N)*(1/fs)

        # create signals
        for i in range(self.nChannels):
            self.x[i,:] = self.amp[i] * np.sin(2*np.pi*self.f[i]*self.t)
            
        return self.t, self.x
    
    def plot(self):
        """
        Plot signals
        """
        
        fig, axs = plt.subplots(self.nChannels, 1)
        fig.subplots_adjust(hspace=0.75)
        
        for i in range(self.nChannels):
            axs[i].plot(self.t, self.x[i,:])
            axs[i].title.set_text(f'Channel {i} - Freq {self.f[i]} Hz')
            axs[i].set_ylabel('amplitude')
            axs[i].set_xlim(0, self.t[-1])
            axs[i].grid()
            
        axs[i].set_xlabel('time [s]') 
        
        plt.show()

if __name__ == "__main__":
    
    nChannels = 2
    fs = 48000
    N = 256
    f = [1000, 2000]
    amp = [1.0, 1.0]
    
    cs = createSignals(amp, N, nChannels)
    t, x = cs.createToneSignals(f, fs)
    cs.plot()