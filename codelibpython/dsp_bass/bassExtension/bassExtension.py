#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bass Extension

@author: G. Howell
"""

import numpy as np

from . import lumpedParam as lp

class bassExtensionParams():
    
    def __init__(self):
        
        pass

    def checkImpedance(self,
                       filename:str):
        
        # TODO implement
        pass
        
    def calcLumpedParams(self,
                         filename:str,
                         voltsPeakAmp:float,
                         Bl:float,
                         Mmc:float,
                         Re:float=None,
                         plot:bool=False,
                         writeToFile:bool=False):
        """ Calculate Lumped Parameters

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
            Electrical resistance [Ohms]. Note, if this is left as None then the resistance is calculated from the lowest
            frequency point from the measured impedance.
        plot : bool, optional
            Plot data. Defaults to False.
        writeToFile : bool
            Write data to file for usage in calculate tuning parameters method.
        """
        
        # import data
        impedData = np.load(filename, allow_pickle=True)
        
        # calculate parameters
        self.lumpParams = lp.calcLumpedParams(impedData['f'], impedData['Z'], voltsPeakAmp, Bl, Mmc, Re, plot=plot)
        
        if writeToFile:
            # TODO
            pass
        
    def calcTuningParams():

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
    
    # calc lumped parameters
    bassExt.calcLumpedParams(impFile, voltsPeakAmp, Bl, Mmc, Re, plot=True, writeToFile=False)
    
    # TODO - run calculate tuning parameters model

    print('\nFinished\n')
