#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

import numpy as np
import scipy.signal as sig

def createFlt2ndOrderZ(fc:float, 
                       Q:float, 
                       fs:int, 
                       warp:bool=True, 
                       filterType:str='lowpass'):
    """ filter2ndOrderZ
    
    Creates a 2nd order discrete time filter designed from the analogue form and the transforming to the discrete
    domain.

    Params
    ======
        fc : float
            Cutoff frequency of the filter [Hz]
        Q : float
            The 'quality' factor of the filter
        fs : int
            Samplerate [Hz]
        warp : bool
            If warp is True, it uses the bilinear transform to transform the coefficient to the discrete domain, else
            it uses the impulse invariance method. Defaults to True.
        filterType : str
            Specifies the type of filter to design, options are 'lowpass', 'highpass' and 'bandpass'. Defaults to 
            'lowpass'.

    Returns
    =======
        bZ : np.array
            Numerator (b) coefficients.
        aZ : np.array
            Numerator (a) coefficients.
    """

    # angular cutoff frequency
    wc = 2*np.pi*fc

    # numerator coefficients
    if (filterType == 'highpass'):
        # set high pass analogue filter response
        #
        #         s^2
        # --------------------
        # s^2 + wc/Q s + wc*wc
        #
    
        bS = np.array([1, 0, 0])
    
    elif (filterType == 'lowpass'):
        # set low pass analogue filter response
        #
        #         wc*wc
        # --------------------
        # s^2 + wc/Q s + wc*wc
        #
    
        bS = np.array([0, 0, wc*wc])
    
    elif (filterType == 'bandpass'):
        # set band pass analogue filter response
        #
        #         s*wc
        # --------------------
        # s^2 + wc/Q s + wc*wc
        #

        bS = np.array([0, wc, 0])
        
    else:
        raise ValueError('Error: invalid filter type')

    # denominator coefficients
    aS = np.array([1, wc/Q, wc*wc])
    
    # convert to discrete domain
    # TODO - create and impinvar function
    if (warp):
        bZ, aZ = sig.bilinear(bS, aS, fs)
    else:
        bZ, aZ = sig.impinvar(bS, aS, fs)
        
    return bZ, aZ

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    # filter params
    fc = 6000
    Q = 0.707
    fs = 24000
    
    # create filter
    b, a = createFlt2ndOrderZ(fc, Q, fs, warp=True, filterType='lowpass')
    
    # plot filter
    fVec = np.linspace(10, fs/2, 100)
    H = sig.freqz(b, a, fVec, fs=fs)[1]

    plt.figure()
    plt.plot(fVec, 20*np.log10(np.abs(H)))
    plt.grid()
    plt.title('Filter Response')
    plt.xlabel('freq [Hz]')
    plt.ylabel('magnitude [dB]')
    plt.xlim(fVec[0], fVec[-1])
    # plt.ylim(-60, 10)
    
    plt.show()
    