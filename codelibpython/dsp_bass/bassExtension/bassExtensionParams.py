#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bass Extension

@author: G. Howell
"""

import numpy as np
import matplotlib.pyplot as plt
from pymatreader import read_mat
import os

import codelibpython.dsp_bass.bassExtension.src as src

class bassExtensionParams():
    
    def __init__(self):
        """ Init
        
        Parameters
        ----------
        """

    def checkImpedance(self,
                       filename:str,
                       plot:bool=True,
                       saveData:bool=False):
        """ Check Impedance

        Parameters
        ----------
        filename : str
            Filename of the impedance file to be loaded. Can be either .mat or .npz file types and the name should
            include the file extension.
        plot : bool, optional
            Plot impedance data. Defaults to True.
        saveData : bool, optional
            Saves the impedance data to a .npz file, appending the original file name with '_impedData'. Defaults to 
            False.
        """
        
        # get the file extension
        fileName, fileExt = os.path.splitext(filename)
        
        # load the data from the file
        try:
            # TODO - think about using a dictionary / lambda form so that all the file type do not have to manually be entered
            # into error function
            match fileExt:
                case '.mat':
                    impedanceData = read_mat(filename)['impedData']
                case '.npz':
                    impedanceData = np.load(filename, allow_pickle=True)
                case _:
                    raise ValueError('Not a recognised file type. Recognised file types are .mat, .npz')
        except OSError as err:
            print("OS Error:", err)
        
        # check impedance data
        self.impedanceData = src.checkImpedance(impedanceData, plot)
        
        # save data
        if saveData:
            np.save(f'{fileName}_impedData', self.impedanceData)
        
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
        
        # get the file extension
        fileName, fileExt = os.path.splitext(filename)
        assert fileExt == '.npy', 'file not of .npy type'
        
        # import data
        impedanceData = np.load(f'{fileName}_impedData{fileExt}', allow_pickle=True)
        
        # calculate parameters
        self.driverParams = src.calcDriverParams(impedanceData.item().get('fVec'), impedanceData.item().get('Himp'), 
                                                 voltsPeakAmp, Bl, Mmc, Re, plot=plot)
        
        # save data
        if saveData:
            np.save(f'{fileName}_driverParams', self.driverParams)
        
    def calcBassExtensionParams(self,
                                filename:str,
                                fcLowExt:float,
                                qExt:float,
                                maxMmPeak:float,
                                maxVoltPeak:float,
                                attackTime:float,
                                releaseTime:float,
                                rmsAttackTime:float,
                                fs:float,
                                dropInd:bool=False,
                                plot:bool=False,
                                saveData:bool=False):

        # get the file extension
        fileName, fileExt = os.path.splitext(filename)
        assert fileExt == '.npy', 'file not of .npy type'
        
        # TODO - import data
        # impedData = np.load(filename, allow_pickle=True)
        
        # calculate bass extension params
        self.bassExtensionParams = src.calcBassExtensionParams(self.driverParams, fcLowExt, qExt, maxMmPeak, 
                                                               maxVoltPeak, attackTime, releaseTime, rmsAttackTime,
                                                               fs, dropInd, plot)
        
        if saveData:
            np.save(f'{fileName}_bassExtParams', self.bassExtensionParams)
        
if __name__ == "__main__":
    
    print('\nCalculating Bass Extension Parameters\n')
    
    # general params
    fs = 48000
    plotData = False
    
    # initialise bass extension
    bassExt = bassExtensionParams()
    
    # check impedance
    bassExt.checkImpedance('impedTestData/01_ALB_IMP_DEQ.npz', plot=plotData, saveData=True)
    
    # measured params
    impFile = 'impedTestData/01_ALB_IMP_DEQ.npy'
    voltsPeakAmp = 17.9 * np.sqrt(2)
    Bl = 5.184
    Mmc = 0.010
    Re = 4.7
    
    # calc driver parameters
    bassExt.calcDriverParams(impFile, voltsPeakAmp, Bl, Mmc, Re, plot=plotData, saveData=True)
    
    # tuning parameters
    driverParamsFile = 'impedTestData/01_ALB_IMP_DEQ.npy'
    fcLowExt = 40
    qExt = 0.65
    maxMmPeak = 1.4
    maxVoltPeak = 20
    attackTime = 0.001
    releaseTime = 0.100
    rmsAttackTime = 0.005
    dropInd = False
    
    # calculate bass extension parameters
    bassExt.calcBassExtensionParams(driverParamsFile, fcLowExt, qExt, maxMmPeak, maxVoltPeak, attackTime, releaseTime,
                                    rmsAttackTime, fs, dropInd, plot=plotData, saveData=True)

    print('\nFinished\n')
