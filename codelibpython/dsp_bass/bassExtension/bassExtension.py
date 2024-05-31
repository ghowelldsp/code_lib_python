#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bass Extension

@author: G. Howell
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

import codelibpython.dsp_filters as dspf
import codelibpython.dsp_maths as dspm

class bassExtension:
    """ Bass Externsion
    
    # TODO - add basic description of module
    """

    def __init__(self,
                 rmsAlpha:float,
                 rmsGainDb:float,
                 rmsGrad:float,
                 rmsYInter:float,
                 smoothAttackCoeff:float,
                 smoothReleaseCoeff:float,
                 poleHigh:float,
                 poleLow:float,
                 extFltSos:np.array,
                 indFltSos:np.array,
                 fs:int,
                 nChannels:int,
                 dtype=np.float32):
        
        # initialise rms params
        self.rmsAlpha = rmsAlpha
        self.rmsGain = dspm.dbToLin(rmsGainDb)
        self.rmsGrad = rmsGrad
        self.rmsYInter = rmsYInter
        
        # initialise smoothing parameters
        self.smoothAttackCoeff = smoothAttackCoeff
        self.smoothReleaseCoeff = smoothReleaseCoeff
        
        self.poleHigh = poleHigh
        self.poleLow = poleLow
        
        # initialise local parameters
        self.fs = fs
        self.nChannels = nChannels
        self.dtype = dtype
        
        # initialise local buffers
        self.poleGainDelayBuff = np.zeros([nChannels])
        
        # initialise filters
        self.extFlt = dspf.biquad(fs, nChannels, extFltSos, dtype=dtype)
        self.indFlt = dspf.biquad(fs, nChannels, indFltSos, dtype=dtype)
        
    def __calcPoleGain(self,
                       x:float):
        """ Calculate Pole Gain
        
        Calculates the pole gain factor that is used to calculate the linearly scaled output of the pole in 
        relationship to the RMS input level. The RMS of the input signal is determined before being linearly scaled to
        a value between 0 and 1 by appling an equation of a straight line where the gradient and y intercept are
        set corresponding to the minimum and maximum RMS levels desired so as to result in the low and high cutoff
        frequency of the applied bass extension filter.
        
        Parameters
        ----------
        x : float
            Input sample.
            
        Returns
        -------
        poleGain : float
            Pole gain factor. Value between 0 and 1.
        """
        
        # determine sample based rms of input signal
        rmsLevel = self.rmsGain * np.sqrt(self.rmsAlpha * x**2)

        # scale factor
        poleGain = rmsLevel * self.rmsGrad + self.rmsYInter

        # limit peak levels
        if (poleGain > 1.0):
            poleGain = 1.0
        elif (poleGain < 0.0):
            poleGain = 0.0
            
        return poleGain
    
    def __calcCoeffs(self,
                     poleGain:float,
                     channel:int):
        """ Calculate Coefficients
        
        Updates the values of the a coefficients of the bass extension filter. Takes the calculated pole gain factor,
        smooths it and then calculates the updated pole locations using a linear interpolation between the locations
        of the poles for the low and high bass extension filters.
        """
        
        poleGainSmooth = self.poleGainDelayBuff[channel]
        
        # selecting smoothing coefficients
        if (poleGain > poleGainSmooth):
            coeff = self.smoothAttackCoeff
        else:
            coeff = self.smoothReleaseCoeff
        
        # weigh input data and running sum
        poleGainSmooth = coeff * (poleGainSmooth - poleGain) + poleGain
        self.poleGainDelayBuff[channel] = poleGainSmooth
        
        # calculate complex poles
        poleLoc = poleGainSmooth * (self.poleHigh - self.poleLow) + self.poleLow
        
        # TODO - check
        val1 = np.conj(gain) * gain
        val2 = -np.conj(gain) - gain
        val3 = 1 + val2.real
        val4 = np.abs(gain) * oklo
        
        # a coefficient vector in the form of [a0, a1, a2]
        aCoeffs = np.array([1, val2.real, val1.real])
        a_1 = val3
        gain = val4
        
        return aCoeffs, a_1, gain
    
    def process(self,
                x:np.array):
        """ Process
        
        Process data through bass extension.

        Parameters
        ----------
        x : np.array [channels][samples]
            Input data.

        Returns
        -------
        y : np.array [channels][samples]
            Output data.
        """
        
        nChannels, nSamples = x.shape
        
        assert (nChannels == self.nChannels), 'Number of channels of input data is not the same as used to init module'
        
        y = np.zeros(x.shape)
        for i in range(nChannels):
            for j in range(nSamples):
                
                xSample = x[i,j]
                
                # calculate pole gain value, 1 = high rms, 0 = low rms
                poleGain = self.__calcPoleGain(xSample)
                
                # calculate the new a coefficients of the bass extension filter
                aCoeffs, a_1, gain = self.__calcCoeffs(poleGain, i)
                
                # set new coefficients
                extFltSos = np.array([self.extFltSos[0,0:3], aCoeffs])
                self.extFlt.setCoeffs(extFltSos)
                
                # process data through extension filter
                ySample = self.extFlt.process(xSample)
                
                # process data through inductance filter
                ySample = self.indFlt.process(ySample)
                
                # assign to output
                y[i,j] = ySample

        return y
    
    def plot(self,
             x:np.array,
             y:np.array):
        """ Plotting
        
        Plots the the inductance filter, and the input vs the output data.

        Parameters
        ----------
        xWet (np.array): 
            2D array of wet input data. In the format [channels][samples]
        y (np.array):
            2D array of output data. In the format [channels][samples]
        """
        
        nChannels, nSamples = x.shape
        
        # time vector
        tVec = np.arange(0,nSamples) * (1/self.fs)
        
        # get frequency reponses
        fVec, Hind = sig.sosfreqz(self.indFltSos, fs=self.fs)
        
        # plot
        fig, axs = plt.subplots(nChannels, 3, squeeze=False)
        fig.subplots_adjust(hspace=0.75)
        fig.suptitle('Input vs. Output', fontsize=14)
        
        # plot inductance
        axs[0,0].semilogx(fVec,  dspm.linToDb(Hind), label='inductance')
        axs[0,0].set_title('Filters')

        for i in range(nChannels):
            axs[i+1,0].plot(tVec, x[i,:], label='input')
            axs[i+1,0].plot(tVec, y[i,:], label='output')
            axs[i+1,0].set_title(f'Input vs Output - Ch{i}')
            axs[i+1,0].set_ylabel('amplitude')
            axs[i+1,0].set_xlim(tVec[0], tVec[-1])
            axs[i+1,0].grid()
        
        axs[-1,0].set_xlabel('time [s]')