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
        
        dxif.designXeqIeqFilters(self.driverParams['bAlign'], 
                                 self.driverParams['aAlign'], 
                                 ftLow,
                                 Qt,
                                 fs,
                                 plotData)
        
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
