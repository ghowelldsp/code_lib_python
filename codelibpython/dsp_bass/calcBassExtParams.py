#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.signal._filter_design as fd

import designXeqIeqFilters as dxif
import designExcursionFilter as defi

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
            'fVec' : driverParams.item().get('fVec'),
            'w0' : driverParams.item().get('w0'),
            'bAlign' : driverParams.item().get('bAlign'),
            'aAlign' : driverParams.item().get('aAlign'),
            'Hdisp' : driverParams.item().get('Hdisp'),
            'HdispGain' : driverParams.item().get('HdispGain'),
            'HdispMm' : driverParams.item().get('HdispMm')
        }
        
        # set measured params
        # TODO - think about moving these to the calculated lumped params script
        self.driverParams['VoltsPeakAmp'] = VoltsPeakAmp
        self.driverParams['Bl'] = Bl
        self.driverParams['Mmc'] = Mmc
        
        # TODO - tidy up
        self.fs = fs
        
    def __writeParams(self):
        
        self.writeParams['extenFlt']['attackAlpha']
        self.writeParams['extenFlt']['attackOneMinusAlpha']
        
    def calcParams(self,
                   ftLow,
                   Qt,
                   maxMmPeak,
                   attackTime,
                   releaseTime,
                   rmsAttackTime,
                   dropIeq,
                   plotData:bool=False):
        
        # self.tuningParams = {
        #     'ftLow' : ftLow,
        #     'Qt' : Qt,
        #     'maxMmPeak' : maxMmPeak,
        #     'attackTime' : attackTime,
        #     'releaseTime' : releaseTime,
        #     'rmsAttackTime' : rmsAttackTime,
        #     'dropIeq' : dropIeq
        # }
        
        # TODO - update plot
        dxif.designXeqIeqFilters(self.driverParams['bAlign'], 
                                 self.driverParams['aAlign'], 
                                 ftLow,
                                 Qt,
                                 fs,
                                 False)
        
        # TODO - impliment
        # % if dropping the IEQ from the main calculation then the IEQ 
        # % response is removed from the overall excursion response and
        # % the coefficients are set to flat
        # if drop_ieq
        #     H_ieqAlignOffset = freqz(bIeq, aIeq, obj.deqParams.alignment.freq, obj.fs);
        #     bIeq = [1 0 0];
        #     aIeq = [1 0 0];
        # else
        #     H_ieqAlignOffset = ones(length(obj.deqParams.alignment.freq),1);
        # end
        
        defi.designExcursionFilter(self.driverParams['fVec'],
                                   self.driverParams['Hdisp'],
                                   self.driverParams['w0'],
                                   self.driverParams['HdispGain'],
                                   self.driverParams['HdispMm'],
                                   filterType='lppeq',
                                   enclosureType='sealed',
                                   fs=fs,
                                   plotData=True)
        
        self.__xeqLimiting():
        
        self.__rms():
            
        self.__saveData():
                    
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
    
    # tuning parameters
    ftLow = 100
    Qt = 0.707
    maxMmPeak = 10
    attackTime = 0.1
    releaseTime = 0.5
    rmsAttackTime = 0.1
    dropIeq = False
    
    # initialise
    # TODO - what is volts peak, etc, doing here? not used?
    lp = calcBassExtParams('codelibpython/dsp_bass/impedTestData/driverParams.npy', VoltsPeakAmp, Bl, Mmc, fs)
    
    # create parameter for a closed box
    params = lp.calcParams(ftLow, Qt, maxMmPeak, attackTime, releaseTime, rmsAttackTime, dropIeq, plotData=True)
    
    print('\nFinished\n')
