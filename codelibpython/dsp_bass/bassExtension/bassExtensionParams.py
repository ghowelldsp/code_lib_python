#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bass Extension

@author: G. Howell
"""

import numpy as np

import src.calcDriverParams as cdp

class bassExtensionParams():
    
    def __init__(self):
        
        pass

    def checkImpedance(self,
                       filename:str):
        
        # TODO implement
        pass
        
    def calcDriverParams(self,
                         filename:str,
                         voltsPeakAmp:float,
                         Bl:float,
                         Mmc:float,
                         Re:float=None,
                         plot:bool=False,
                         writeToFile:bool=False):
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
        writeToFile : bool
            Write data to file for usage in calculate tuning parameters method.
        """
        
        # import data
        impedData = np.load(filename, allow_pickle=True)
        
        # calculate parameters
        self.driverParams = cdp.calcDriverParams(impedData['f'], impedData['Z'], voltsPeakAmp, Bl, Mmc, Re, plot=plot)
        
        if writeToFile:
            # TODO
            pass
        
    def calcBassExtensionParams():

        # TODO - implement
        pass
    
if __name__ == "__main__":
    
    print('\nCalculating Bass Extension Parameters\n')
    
    # TODO - run check impedance method
    
    # initialise bass extension
    bassExt = bassExtensionParams()
    
    # measured params
    impFile = "impedTestData/01_ALB_IMP_DEQ_reformatted.npz"
    voltsPeakAmp = 17.9 * np.sqrt(2)
    Bl = 5.184
    Mmc = 0.010
    Re = 4.7
    
    # calc driver parameters
    bassExt.calcDriverParams(impFile, voltsPeakAmp, Bl, Mmc, Re, plot=True, writeToFile=False)
    
    # calculate bass extension parameters
    bassExt.calcBassExtensionParams()

    print('\nFinished\n')
