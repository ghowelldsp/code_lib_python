#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.signal._filter_design as fd

class calcBassExtParams():
    
    def __init__(self,
                 driverParamsFileName,
                 VoltsPeakAmp,
                 Bl,
                 Mmc,
                 fs):
        
        # load driver params
        driverParams = np.load(driverParamsFileName, allow_pickle=True)
        
        # load variables
        self.driverParams = {
            'bAlign' : driverParams.item().get('bAlign'),
            'aAlign' : driverParams.item().get('aAlign')
        }
        
        # set measured params
        # TODO - think about moving these to the calculated lumped params script
        self.driverParams['VoltsPeakAmp'] = VoltsPeakAmp
        self.driverParams['Bl'] = Bl
        self.driverParams['Mmc'] = Mmc
        
        # TODO - tidy up
        self.fs = fs
        
    def __designXeqIeqFilters(self,
                              tuningParams):
        
        # calculate discrete coefficients using bilinear transform
        bZ, aZ = sig.bilinear(self.driverParams['bAlign'], self.driverParams['aAlign'], self.fs)
        
        # determine pole zero locations
        z, p, k1 = sig.tf2zpk(bZ, aZ)
        
        # find and remove the zero relating to the 1st order lowpass filter
        z = fd._cplxpair(z)
        idx = np.where(np.real(z) == z)[0]
        if (idx.size != 0):
            idx = idx[0]
        if (np.iscomplex(z[idx]) or (z[idx] > -0.9) or (z[idx] > -1.1)):
            # TODO - handle this error properly 
            print('Error!')
        zLp1 = np.array([z[idx]])
        z = np.delete(z, idx)
        
        # find and remove the pole relating to the 1st order lowpass filter
        p = fd._cplxpair(p)
        idx = np.where(np.real(p) == p)[0]
        if (idx.size != 0):
            idx = idx[0]
        if (np.iscomplex(p[idx])):
            # TODO - handle this error properly 
            print('Error!')
        pLp1 = np.array([p[idx]])
        p = np.delete(p, idx)

        # check for a 3rd order sytem
        if (len(z) != 3) or (len(p) != 3):
            # TODO - handle this error properly 
            print('Error!')                                                        	

        # calculate coefficient values of the 2nd order lowpass filter with normalised gain
        bLp2, aLp2 = sig.zpk2tf(z, p, k=1)
        
        # plot data
        if self.plotData:
            
            # create frequency vector and alignment transfer functions
            fVec = np.linspace(10, self.fs/2, 1000)
            Hcont = sig.freqs(self.driverParams['bAlign'], self.driverParams['aAlign'], 2*np.pi*fVec)[1]
            Hdisc = sig.freqz(bZ, aZ, fVec, fs=fs)[1]
            HLp2 = sig.freqz(bLp2, aLp2, fVec, fs=fs)[1]
            bLp1, aLp1 = sig.zpk2tf(zLp1, pLp1, k=1)
            HLp1 = k1 * sig.freqz(bLp1, aLp1, fVec, fs=fs)[1]
            
            # plot data
            plt.figure()
            plt.semilogx(fVec, 20*np.log10(Hcont), label='continuous')
            plt.semilogx(fVec, 20*np.log10(Hdisc), label='discrete')
            plt.semilogx(fVec, 20*np.log10(HLp1), label='1st order LP')
            plt.semilogx(fVec, 20*np.log10(HLp2), label='2nd order LP')
            plt.legend()
            plt.grid()
            plt.title('Alignment')
            plt.xlabel('freq [Hz]')
            plt.ylabel('magnitude [dB]')
            plt.xlim(fVec[0], fVec[-1])
            plt.ylim(-60, 10)
        
    def calcParams(self,
                   ftLow,
                   Qt,
                   maxMmPeak,
                   attackTime,
                   releaseTime,
                   rmsAttackTime,
                   dropIeq,
                   plotData:bool=False):
        
        self.tuningParams = {
            'ftLow' : ftLow,
            'Qt' : Qt,
            'maxMmPeak' : maxMmPeak,
            'attackTime' : attackTime,
            'releaseTime' : releaseTime,
            'rmsAttackTime' : rmsAttackTime,
            'dropIeq' : dropIeq
        }
        
        self.plotData = plotData
        
        self.__designXeqIeqFilters(self.tuningParams)
        
        if plotData:
            plt.show()
    
if __name__ == "__main__":
    
    print('\nCalculating Bass Extension Parameters\n')
    
    # general parameters
    fs = 48000
    
    # measured params
    VoltsPeakAmp = 15
    Bl = 10
    Mmc = 0.010
    Re = 4.0
    
    # tuning parameters
    ftLow = 100
    Qt = 0.707
    maxMmPeak = 10
    attackTime = 0.1
    releaseTime = 0.5
    rmsAttackTime = 0.1
    dropIeq = False
    
    # initialise
    lp = calcBassExtParams('impedTestData/driverParams.npy', VoltsPeakAmp, Bl, Mmc, fs)
    
    # create parameter for a closed box
    params = lp.calcParams(ftLow, Qt, maxMmPeak, attackTime, releaseTime, rmsAttackTime, dropIeq, plotData=True)
    
    print('\nFinished\n')
