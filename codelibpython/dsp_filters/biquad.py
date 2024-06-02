# -*- coding: utf-8 -*-
"""
Biquad Filter

@author: G. Howell
"""

from scipy import signal as sig
import numpy as np
import matplotlib.pyplot as plt

import codelibpython.dsp_maths as dspm
import codelibpython.dsp_utils as dspu

## BIQUAD CLASS #######################################################################################################

class biquad:
    """
    Biquad Filter
    
    Implimentation of a biquad filter using various models.
    """

    def __init__(self, 
                 fs:int,
                 nChannels:int,
                 nStages:int,
                 sos:np.array=None,
                 modelType:str='scipy',
                 dtype:np.dtype=np.float32):
        """ Init
        
        Parameters
        ----------
        fs : int    
            Sample rate [Hz].
        nChannels : int
            Number of input / output channels.
        sos : np.array [nStages][6]
            SOS matrix of coefficients in the form of [b0, b1, b2, a0, a1, a2].
        modelType : string  
            String of biquad structure to be used. Options are either 'scipy', 'df1', 'df2' or 'df2t'.
        dtype : np.dtype
            Numpy datatype
        """
        
        self.fs = fs
        self.nChannels = nChannels
        self.nStages = nStages
        self.modelType = modelType
        self.dtype = dtype
        if np.any(sos != None):
            self.setCoeffs(sos)
            
        # create empty delay registers to enable block processing
        if (modelType == 'scipy') or (modelType == 'df2') or (modelType == 'df2t'):
            self.delayReg = np.zeros((nStages, nChannels, 2), dtype)
        elif modelType == 'df1':
            self.delayReg = np.zeros((nStages, nChannels, 4), dtype)
        
    def createCoeffs(self, 
                     order:np.array, 
                     fc:np.array, 
                     fltType:str='low'):    
        """ Create Coefficients
        
        Create filter coefficients for the biquad which are in the form of sos sections.
        
        Parameters
        ----------
        order : int
            Order of filter
        fc : int        
            Cutoff frequency
        fltType : list
            Filter type
        """
        
        # create butterworth filters
        sos = sig.butter(order, fc, btype=fltType, output='sos', fs=self.fs)
        
        # set the coefficients
        self.setCoeffs(sos)
        
    def setCoeffs(self,
                  sos:np.array):
        """ Set Coefficients

        Parameters
        ----------
        x : np.array [channels][samples]
            Input data.
        sos : np.array [stages][6]
            SOS matrix of coefficients where each row has the coefficients ordering in the form of 
            [b0, b1, b2, a0, a1, a2].
        """
        
        assert sos.ndim == 2, 'SOS matrix is not the correct number of dimensions'
        assert sos.shape[0] == self.nStages, 'SOS matrix number of stages is less than init'
        assert sos.shape[1] == 6, 'SOS matrix does not contain the correct number of coefficients'
        
        # set the coefficients
        self.sos = sos.astype(self.dtype)
        
    def process(self, 
                x:np.array):
        """ Process
        
        Processes input data though the filter, returning the processed output data.
        
        Parameters
        ----------
        x : np.array [channels][samples]
            Input data.
        
        Returns
        -------
        y : np.array [channels][samples]        
            Output data.
        """
        
        assert x.ndim == 2, 'input data is not two dimensional'
        assert x.shape[0] == self.nChannels, 'input data is not the same number of channels as initialisation'
        
        # convert to specified datatype
        x.astype(self.dtype)
        y = np.zeros(x.shape, dtype=self.dtype)
        
        # get parameters
        _, nSamples = x.shape
        nStages = self.sos.shape[0]
        
        if (self.modelType == "scipy"):
            # filter using scipy's native function
            y, self.delayReg = sig.sosfilt(self.sos, x, axis=1, zi=self.delayReg)
                
        else:
            # filter with direct form 1 structure
            
            xTmp = np.array(x)
                
            for i in range(nStages):
                
                # load coefficients
                b0 = self.sos[i,0]
                b1 = self.sos[i,1]
                b2 = self.sos[i,2]
                a1 = -self.sos[i,4]
                a2 = -self.sos[i,5]
                
                if self.modelType == "df1":
                
                    # load delay values
                    dx1 = self.delayReg[i,:,0]
                    dx2 = self.delayReg[i,:,1]
                    dy1 = self.delayReg[i,:,2]
                    dy2 = self.delayReg[i,:,3]
                    
                    for j in range(nSamples):
                        
                        xn = xTmp[:,j]
                        
                        yn = b0 * xn + b1 * dx1 + b2 * dx2
                        yn += a1 * dy1 + a2 * dy2
                        
                        dx2 = dx1
                        dx1 = xn
                        
                        dy2 = dy1
                        dy1 = yn
                        
                        y[:,j] = yn
                    
                    # save delay values to object
                    self.delayReg[i,:,0] = dx1
                    self.delayReg[i,:,1] = dx2
                    self.delayReg[i,:,2] = dy1
                    self.delayReg[i,:,3] = dy2
                    
                elif self.modelType == "df2":
                    
                    # TODO  - impliment
                    pass
                
                    # # load delay values
                    # d1 = self.delayReg[j,0,i]
                    # d2 = self.delayReg[j,1,i]
                    
                    # # tmp value
                    # dn = np.zeros([1], dtype=dtype)
                
                    # for k in range(N):
                        
                    #     xn = xTmp[i,k]
                        
                    #     dn = xn + a1 * d1 + a2 * d2
                    #     yn = b0 * dn + b1 * d1 + b2 * d2
                        
                    #     d2 = d1
                    #     d1 = dn
                        
                    #     y[i,k] = yn
                    
                    # # save delay values to object
                    # self.delayReg[j,0,i] = d1
                    # self.delayReg[j,1,i] = d2
                
                elif self.modelType == "df2t":
                    
                    # TODO  - impliment
                    pass
                
                    # # load delay values
                    # d1 = self.delayReg[j,0,i]
                    # d2 = self.delayReg[j,1,i]
                
                    # for k in range(N):
                        
                    #     xn = xTmp[i,k]
                        
                    #     yn = b0 * xn + d1
                        
                    #     d1 = b1 * xn + d2
                    #     d1 += a1 * yn
                        
                    #     d2 = b2 * xn
                    #     d2 += a2 * yn
                        
                    #     y[i,k] = yn
                    
                    # # save delay values to object
                    # self.delayReg[j,0,i] = d1
                    # self.delayReg[j,1,i] = d2
                    
                # update input to next stage
                xTmp = y
        
        return y
    
    def plot(self, 
             x:np.array,
             y:np.array):
        """ Plot Response
        
        Plots the frequency response of filter.
        
        Parameters
        ----------
        x : np.array [channels][samples]
            Input data.
        y : np.array [channels][samples]
            Output data.
        """
        
        assert x.shape == y.shape, 'input and output data are not of same shape'
        
        # get number of channels and samples in data
        nChannels, nSamples = x.shape
        
        # time vector
        tVec = np.arange(0,nSamples) * (1/self.fs)
        
        # calculate frequency response
        fVec, H = sig.sosfreqz(self.sos, fs=self.fs)
        
        fig, ax = plt.subplots(nChannels + 1, 1, squeeze=False)
        fig.subplots_adjust(hspace=0.75)
        
        # plot filter repsonse
        ax[0,0].semilogx(fVec, dspm.linToDb(H))
        ax[0,0].grid()
        ax[0,0].set_title(f'Frequency Response - Filter')
        ax[0,0].set_xlabel('freq [Hz]')
        ax[0,0].set_ylabel('magnitude [dB]')
        ax[0,0].set_xlim(fVec[0], self.fs/2)
        
        # plot time domain
        for i in range(nChannels):
            ax[i+1,0].plot(t, x[i,:], label='input')
            ax[i+1,0].plot(t, y[i,:], label='output')
            ax[i+1,0].grid()
            ax[i+1,0].set_title(f'Input vs Output - Channel {i}')
            ax[i+1,0].set_ylabel('amplitude')
            ax[i+1,0].set_xlim(0, t[-1])
            
        ax[i+1,0].set_xlabel('time [s]')
        ax[i+1,0].legend()

if __name__ == "__main__":
    
    # model parameters
    dtype = np.float32
    
    # input signal settings
    nChannels = 3
    amp = np.full(nChannels, 1.0)
    freq = np.array([1000, 2000, 3000])
    fs = 48000
    blockLen = 512
    nBlocks = 2
    N = blockLen * nBlocks
    
    # create input signals
    x, t = dspu.createToneSignals(amp, freq, N, nChannels, fs, dtype)
        
    # initalise biquad
    biquad = biquad(fs, nChannels, nStages=1, modelType='df1', dtype=dtype)
    
    # create coefficients
    biquad.createCoeffs(order=2, fc=1000, fltType='low')
    
    # process data
    y = np.zeros(x.shape, dtype=dtype)
    for i in range(nBlocks):
        indexRange = np.arange((i * blockLen), ((i+1) * blockLen))
        y[:,indexRange] = biquad.process(x[:,indexRange])
    
    # plot data
    biquad.plot(x, y)       # plots the input signals vs model output signals
    plt.show()
    