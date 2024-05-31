    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

import matplotlib.pyplot as plt
import numpy as np
from pymatreader import read_mat
import os

def checkImpedance(filename:str, 
                   plot:bool=True,
                   saveData:bool=False,
                   dtype:np.dtype=np.float32):
    """ Check Impedance
    
    Enable checking on impedance data via a visual inspection of the impedance curve. Loads the data from either a .mat
    or .npz file format, which contains the frequency vector, magnitude and phase. A complex impedance vector is 
    calculated and, if selected, saved to file along with the frequency vector for use in calculating the driver parameters.

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

    Returns
    -------
    impedanceData : dict
        fVec : np.array [bins]
            Frequency vector.
        Himp : np.array [bins]
            Complex impedance transfer function.
    """
    
    # get the file extension
    fileName, fileExt = os.path.splitext(filename)
    
    # load the data from the file
    try:
        # TODO - think about using a dictionary / lambda form so that all the file type do not have to manually be entered
        # into error function
        match fileExt:
            case '.mat':
                data = read_mat(filename)['impedData']
            case '.npz':
                data = np.load(filename, allow_pickle=True)
            case _:
                raise ValueError('Not a recognised file type. Recognised file types are .mat, .npz')
    except OSError as err:
        print("OS Error:", err)
    
    # get data
    try:
        fVec = np.asarray(data['freq'], dtype=dtype)
        mag = np.asarray(data['mag'], dtype=dtype)
        phase = np.asarray(data['phase'], dtype=dtype)
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
    
    # save data
    if saveData:
        np.savez(fileName, fVec=fVec, Himp=Himp)
    
    if plot:
        plt.figure()
        
        plt.subplot(2,1,1)
        plt.plot(fVec, mag)
        plt.grid()
        plt.title('Impedance Magnitude')
        plt.ylabel('Mag [Ohm]')
        plt.xlim(fVec[0], fVec[-1])
        
        plt.subplot(2,1,2)
        plt.plot(fVec, phase)
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
