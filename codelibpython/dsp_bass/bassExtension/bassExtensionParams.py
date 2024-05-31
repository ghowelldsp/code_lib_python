#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bass Extension

@author: G. Howell
"""

import numpy as np

import src

class bassExtensionParams():
    
    def __init__(self,
                 fs,
                 dtype:np.dtype=np.float32):
        """ Init

        """

    def checkImpedance(self,
                       filename:str,
                       plot:bool=True,
                       saveData:bool=False):
        """ Check Impedance

        Parameters
        ----------
        filename : str
            Filename of the impedance file to be loaded. Can be either .mat or .npz file types.
        plot : bool, optional
            Plot impedance data. Defaults to True.
        saveData : bool, optional
            Saves the impedance data to a .npz file. Defaults to False.
        dtype : np.dtype, optional
            Datatype. Defaults to np.float32.
        """
        
        self.impedanceData = src.checkImpedance(filename, plot, saveData, self.dtype)
        
        # TODO - move the save data feature to this class
        
    def calcDriverParams(self,
                         filename:str,
                         voltsPeakAmp:float,
                         Bl:float,
                         Mmc:float,
                         Re:float=None,
                         plot:bool=False,
                         saveData:bool=False):
        """ Calculate Driver Parameters

        Parameters
        ----------
        filename : str
            Filename of the impedance data.
        voltsPeakAmp : float
            Voltage at peak amplitude [V].
        Bl : float
            Force factor.
        Mmc : float
            Moving mass [g].
        Re : float
            Electrical resistance [Ohms]. Note, if this is left as None then the resistance is calculated from the 
            lowest frequency point from the measured impedance.
        plot : bool, optional
            Plot data. Defaults to False.
        saveData : bool
            Save data to file for usage in calculate tuning parameters method.
        """
        
        # TODO - add a check to see if impedance data has been created or if we need to load from a file.
        
        # import data
        impedData = np.load(filename, allow_pickle=True)
        
        # calculate parameters
        self.driverParams = src.calcDriverParams(impedData['fVec'], impedData['Himp'], voltsPeakAmp, Bl, Mmc, Re, plot=plot)
        
        if saveData:
            # TODO
            pass
        
    def calcBassExtensionParams(self,
                                fcLowExt:float,
                                qExt:float,
                                maxMmPeak:float,
                                maxVoltPeak:float,
                                attackTime:float,
                                releaseTime:float,
                                rmsAttackTime:float,
                                dropInd:bool=False,
                                plot:bool=False,
                                saveData:bool=False):

        # TODO - import data
        # impedData = np.load(filename, allow_pickle=True)
        
        # calculate bass extension params
        self.bassExtensionParams = src.calcBassExtensionParams(self.driverParams, fcLowExt, qExt, maxMmPeak, 
                                                               maxVoltPeak, attackTime, releaseTime, rmsAttackTime,
                                                               self.fs, dropInd, plot)
        
        if saveData:
            # TODO
            pass
        
if __name__ == "__main__":
    
    print('\nCalculating Bass Extension Parameters\n')
    
    # general params
    fs = 48000
    plot = True
    
    # initialise bass extension
    bassExt = bassExtensionParams(fs)
    
    # TODO - run check impedance method
    
    # measured params
    impFile = "impedTestData/01_ALB_IMP_DEQ_reformatted.npz"
    voltsPeakAmp = 17.9 * np.sqrt(2)
    Bl = 5.184
    Mmc = 0.010
    Re = 4.7
    
    # calc driver parameters
    bassExt.calcDriverParams(impFile, voltsPeakAmp, Bl, Mmc, Re, plot=plot, writeToFile=False)
    
    # tuning parameters
    fcLowExt = 40
    qExt = 0.65
    maxMmPeak = 1.4
    maxVoltPeak = 20
    attackTime = 0.001
    releaseTime = 0.100
    rmsAttackTime = 0.005
    dropInd = False
    
    # calculate bass extension parameters
    bassExt.calcBassExtensionParams(fcLowExt, fcLowExt, qExt, maxMmPeak, maxVoltPeak, attackTime, releaseTime, 
                                    rmsAttackTime, dropInd, plot)

    print('\nFinished\n')
