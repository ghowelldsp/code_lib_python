#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

import matplotlib.pyplot as plt
import numpy as np

def checkImpedance(impedanceData:str, 
                   plot:bool=True):
    """ Check Impedance
    
    Enables checking on impedance data via a visual inspection of the impedance curve. Takes in the measured impedance
    magnitude and phase, including the respective measurement frequency vector. Return the complex impedance data and
    associated frequency vector.

    Parameters
    ----------
    impedanceData : dict
        freq : np.array
            Vector of frequency points of the measured impedance [Hz].
        mag : np. array
            Measured impedance magnitude vector [Ohms].
        phase : np.array
            Measured impedance phase vector [degrees].
    plot : bool, optional
        Plot impedance data. Defaults to True.

    Returns
    -------
    impedanceData : dict
        fVec : np.array [bins]
            Frequency vector [Hz].
        Himp : np.array [bins]
            Complex impedance transfer function [Ohms].
    """
    
    # get data
    try:
        fVec = np.asarray(impedanceData['freq'], dtype=np.float64)
        mag = np.asarray(impedanceData['mag'], dtype=np.float64)
        phase = np.asarray(impedanceData['phase'], dtype=np.float64)
    except KeyError:
        print('frequency, magnitude or phase vector not present in data')
    
    assert ((len(fVec) == len(mag)) or (len(fVec) == len(phase))), 'freq data is not the same length as the magnitude \
                                                                     and / or phase'
    
    # create complex impedance vector
    Himp = mag * np.exp(1j * (phase * np.pi/180.0))
    
    # create return data structure
    impedanceData = {
        'fVec' : fVec,
        'Himp' : Himp
    }
    
    if plot:
        plt.figure()
        
        plt.subplot(2,1,1)
        plt.semilogx(fVec, mag)
        plt.grid()
        plt.title('Impedance Magnitude')
        plt.ylabel('Mag [Ohm]')
        plt.xlim(fVec[0], fVec[-1])
        
        plt.subplot(2,1,2)
        plt.semilogx(fVec, phase)
        plt.grid()
        plt.title('Impedance Phase')
        plt.ylabel('Phase [deg]')
        plt.xlabel('freq [Hz]')
        plt.xlim(fVec[0], fVec[-1])
        
        plt.show()
    
    return impedanceData

if __name__ == "__main__":
    
    filename = "../impedTestData/testData.mat"
    
    impedanceData = checkImpedance(filename, saveData=True)
