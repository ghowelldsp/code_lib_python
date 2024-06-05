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
                 driverParams:dict,
                 nChannels:int,
                 dtype=np.float32):
        """ Init

        # TODO
        Args:
            driverParams (dict): _description_
            nChannels (int): _description_
            dtype (_type_, optional): _description_. Defaults to np.float32.
        """
        
        # initialise rms params
        self.rmsAttackCoeff = driverParams['rmsAttackCoeff']
        self.rmsGain = driverParams['rmsGain']
        self.rmsGrad = driverParams['rmsGrad']
        self.rmsYInter = driverParams['rmsYInter']
        
        # initialise smoothing parameters
        self.smoothAttackCoeff = driverParams['smoothAttackCoeff']
        self.smoothReleaseCoeff = driverParams['smoothReleaseCoeff']
        
        self.poleHigh = driverParams['poleHigh']
        self.poleLow = driverParams['poleLow']
        
        # TODO - need to look at these in more detail
        self.kLow = driverParams['kLow']
        self.kLowInv = driverParams['kLowInv']
        
        # initialise local parameters
        self.fs = driverParams['fs']
        self.nChannels = nChannels
        self.dtype = dtype
        
        # initialise local buffers
        self.msVal = np.ones([nChannels])
        self.poleGain = np.ones([nChannels])
        
        # initialise filters
        self.bqExt = dspf.biquad(driverParams['fs'], nChannels, 1, driverParams['sosExtHigh'], dtype=dtype)
        self.bCoeffsExt = driverParams['sosExtHigh'][0,0:3]
        self.bqInd = dspf.biquad(driverParams['fs'], nChannels, 1, driverParams['sosInd'], dtype=dtype)
        
        self.driverParams = driverParams
        
    def __weightedSum(self,
                      alpha,
                      x1,
                      x2):
        """ Weighted Sum
        
        TODO

        Args:
            alpha (_type_): _description_
            x1 (_type_): _description_
            x2 (_type_): _description_

        Returns:
            _type_: _description_
        """
        # y = alpha * x1 + (1 - alpha) * x2
        # y = alpha * (x1 - x2) + x2
        
        return alpha * (x1 - x2) + x2
        
    def __calcPeakLevel(self,
                        x:np.array):
        """ Calculate Peak Level

        Calculates the smoothed RMS level then applies a sqrt(2) gain to convert to peak level.
        
        Parameters
        ----------
        x : np.array [channels]
            Peak level.
            
        Returns
        -------
        peakLevel : np.array [channels]
            Peak level.
        """
        
        # determine sample based rms (really mean square value) of input signal with the output smoothed
        self.msVal = self.__weightedSum(self.rmsAttackCoeff, self.msVal, x**2)
        
        # calc rms and apply to convert to peak level
        peakLevel = self.rmsGain * np.sqrt(self.msVal)
        
        return peakLevel
    
    def __calcPoleGain(self,
                       peakLevel:np.array):
        """ Calculate Pole Gain

        Calculates the normalised gain used to determine the location of the poles.
        
        Parameters
        ----------
        peakLevel : np.array [channels]
            Peak level.
            
        Returns
        -------
        poleGain : np.array [channels]
            Pole gain, normalised value between 0 and 1.
        """
        
        # scale factor
        poleGain = peakLevel * self.rmsGrad + self.rmsYInter

        # limit
        poleGain = np.where(poleGain > 1.0, 1.0, poleGain)
        poleGain = np.where(poleGain < 0.0, 0.0, poleGain)
        
        # select smoothing coefficients
        coeffs = np.where(poleGain > self.poleGain, self.smoothAttackCoeff, self.smoothReleaseCoeff)
        
        # smooth pole gain value
        self.poleGain = self.__weightedSum(coeffs, self.poleGain, poleGain)
        
        return poleGain
    
    def __calcCoeffs(self,
                     poleGain:np.array):
        """ Calculate Coefficients
        
        Updates the values of the a coefficients of the bass extension filter. Takes the calculated pole gain factor,
        smooths it and then calculates the updated pole locations using a linear interpolation between the locations
        of the poles for the low and high bass extension filters.
        """
        
        # calculate complex poles
        poleLoc = self.__weightedSum(poleGain, self.poleHigh, self.poleLow)
        
        # TODO - tidy up
        a1 = -2*poleLoc.real
        a2 = np.conj(poleLoc) * poleLoc
        a_1 = poleGain + a1.real
        gain = np.abs(poleLoc) * self.kLowInv
        
        # a coefficient vector in the form of [a0, a1, a2
        aCoeffs = np.array([1.0, a1[0], a2[0].real])
        
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
        for i in range(nSamples):
            
            xSample = x[:,i]
            
            # calculate smoothed rms level
            peakLevel = self.__calcPeakLevel(xSample)
        
            # calculate smoothed pole gain value, 1 = high rms, 0 = low rms
            poleGain = self.__calcPoleGain(peakLevel)
        
            # calculate coefficients
            aCoeffsExt, _, _ = self.__calcCoeffs(poleGain)
            
            # set new coefficients
            self.sosExtCurr = np.array(np.concatenate([self.bCoeffsExt, aCoeffsExt]), ndmin=2)
            self.bqExt.setCoeffs(self.sosExtCurr)
        
            # process data through extension filter
            ySample = self.bqExt.process(xSample[:,np.newaxis])
            
            # process data through inductance filter
            ySample = self.bqInd.process(ySample)
            
            # assign to output
            y[:,i] = ySample

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
        # TODO - create a log fVec
        fVec, Hind = sig.sosfreqz(self.driverParams['sosInd'], worN=4096, fs=self.fs)
        fVec, HextLow = sig.sosfreqz(self.driverParams['sosExtLow'], worN=4096, fs=self.fs)
        fVec, HextHigh = sig.sosfreqz(self.driverParams['sosExtHigh'], worN=4096, fs=self.fs)
        fVec, HextCurr = sig.sosfreqz(self.sosExtCurr, worN=4096, fs=self.fs)
        
        # plot
        fig, axs = plt.subplots(nChannels + 1, 1, squeeze=False)
        fig.subplots_adjust(hspace=0.75)
        fig.suptitle('Input vs. Output', fontsize=14)
        
        # plot inductance
        # axs[0,0].semilogx(fVec, dspm.linToDb(Hind), label='inductance')
        axs[0,0].semilogx(fVec, dspm.linToDb(HextCurr), label='Current')
        axs[0,0].semilogx(fVec, dspm.linToDb(HextLow), '--', label='Low RMS')
        axs[0,0].semilogx(fVec, dspm.linToDb(HextHigh), '--', label='High RMS')
        axs[0,0].grid()
        axs[0,0].set_title('Extension Filter')
        axs[0,0].set_xlim(fVec[0], fVec[-1])
        axs[0,0].legend()

        for i in range(nChannels):
            axs[i+1,0].plot(tVec, x[i,:], label='input')
            axs[i+1,0].plot(tVec, y[i,:], label='output')
            axs[i+1,0].set_title(f'Input vs Output - Ch{i}')
            axs[i+1,0].set_ylabel('amplitude')
            axs[i+1,0].set_xlim(tVec[0], tVec[-1])
            axs[i+1,0].grid()
        
        axs[-1,0].set_xlabel('time [s]')
        
        plt.show()
        
if __name__ == "__main__":
    
    import codelibpython.dsp_utils as dspu
    
    print('\nCalculating Bass Extension\n')
    
    # input signal params
    # TODO - need to setup the biquad for more than one channels
    nSignals = 1
    fs = 48000
    N = 1024 * 10
    f = [65, 2000]
    amp = [0.01, 1.0]
    
    # create an input signal
    x, t = dspu.createToneSignals(amp, f, N, nSignals, fs)
    
    # load bass extension params
    bassExtParamsTmp = np.load('impedTestData/01_ALB_IMP_DEQ_bassExtParams.npy', allow_pickle=True)
    bassExtParams = {
        'rmsAttackCoeff' : bassExtParamsTmp.item().get('rmsAttackCoeff'),
        'rmsGain' : bassExtParamsTmp.item().get('rmsGain'),
        'rmsGrad' : bassExtParamsTmp.item().get('rmsGrad'),
        'rmsYInter' : bassExtParamsTmp.item().get('rmsYInter'),
        'smoothAttackCoeff' : bassExtParamsTmp.item().get('smoothAttackCoeff'),
        'smoothReleaseCoeff' :  bassExtParamsTmp.item().get('smoothReleaseCoeff'),
        'poleHigh' : bassExtParamsTmp.item().get('poleHigh'),
        'poleLow' : bassExtParamsTmp.item().get('poleLow'),
        'sosExtLow' : bassExtParamsTmp.item().get('sosExtLow'),
        'sosExtHigh' : bassExtParamsTmp.item().get('sosExtHigh'),
        'sosInd' : bassExtParamsTmp.item().get('sosInd'),
        'fs' : fs,
        'kLow' : bassExtParamsTmp.item().get('kLow'),
        'kLowInv' : bassExtParamsTmp.item().get('kLowInv'),
    }
    
    # initialise
    bassExt = bassExtension(bassExtParams, nSignals)
    
    # process signal
    y = bassExt.process(x)
    
    # plot
    bassExt.plot(x,y)
    
    print('\nFinished\n')
