#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

import numpy as np
import scipy.signal as sig
import scipy.optimize as opt
import matplotlib.pyplot as plt

import codelibpython.dsp_maths as dspm

def _fitToSealed(paramsEst:np.array,
                 filterType,
                 fVec,
                 excurNorm,
                 fs):
    
    # load variables
    # fs = paramsFix['fs']
    # fVec = paramsFix['fVec']
    # excurNorm = paramsFix['excurNorm']
    
    match filterType:
        
        case 'lp':
            
            # get values 
            gain = paramsEst[0]
            lpFc = paramsEst[1]
            lpQ = paramsEst[2]
            
            # check limits
            if lpFc < 10:
                lpFc = 10
            if lpQ < 0.1:
                lpQ = 0.1
            
            # calc lowpass coefficients
            lpB, lpA = dspm.createFlt2ndOrderZ(lpFc, lpQ, fs, filterType='lowpass')
            
            # form sos matrix
            sos = np.zeros([1,6])
            sos[0,0:3] = lpB
            sos[0,3:6] = lpA
            
        case 'lppeq':
            
            # get values 
            gain = paramsEst[0]
            lpFc = paramsEst[1]
            lpQ = paramsEst[2]
            peqFc = paramsEst[3]
            peqGain = paramsEst[4]
            peqQ = paramsEst[5]
            
            # check limits
            if lpFc < 10:
                lpFc = 10
            if lpQ < 0.1:
                lpQ = 0.1
            if peqFc < 10:
                peqFc = 10
            if peqQ < 0.1:
                peqQ = 0.1
            
            # calc lowpass coefficients
            lpB, lpA = dspm.createFlt2ndOrderZ(lpFc, lpQ, fs, filterType='lowpass')
            
            # calc peq coefficients
            peqB, peqA = dspm.parametricEq(peqFc, peqQ, peqGain, fs, peqFc);
            
            # form sos matrix
            sos = np.zeros([2,6])
            sos[0,0:3] = lpB
            sos[0,3:6] = lpA
            sos[1,0:3] = peqB
            sos[1,3:6] = peqA
        
        case 'lp2peq':
            
            # get values 
            gain = paramsEst[0]
            lpFc = paramsEst[1]
            lpQ = paramsEst[2]
            peq1Fc = paramsEst[3]
            peq1Gain = paramsEst[4]
            peq1Q = paramsEst[5]
            peq2Fc = paramsEst[6]
            peq2Gain = paramsEst[7]
            peq2Q = paramsEst[8]
            
            # check limits
            if lpFc < 10:
                lpFc = 10
            if lpQ < 0.1:
                lpQ = 0.1
            if peq1Fc < 10:
                peq1Fc = 10
            if peq1Q < 0.1:
                peq1Q = 0.1
            if peq2Fc < 10:
                peq2Fc = 10
            if peq2Q < 0.1:
                peq2Q = 0.1
            
            # calc lowpass coefficients
            lpB, lpA = dspm.createFlt2ndOrderZ(lpFc, lpQ, fs, filterType='lowpass')
            
            # calc peq coefficients
            peq1B, peq1A = dspm.parametricEq(peq1Fc, peq1Q, peq1Gain, fs)
            peq2B, peq2A = dspm.parametricEq(peq2Fc, peq2Q, peq2Gain, fs)
            
            # form sos matrux
            sos = np.array([[lpB, lpA], [peq1B, peq1A], [peq2B, peq2A]])
            
            # form sos matrix
            sos = np.zeros([3,6])
            sos[0,0:3] = lpB
            sos[0,3:6] = lpA
            sos[1,0:3] = peq1B
            sos[1,3:6] = peq1A
            sos[2,0:3] = peq2B
            sos[2,3:6] = peq2A
            
    # calc transfer function of filters
    H = gain * sig.sosfreqz(sos, fVec, fs)[1]

    # calculate weights                          
    weights = np.abs(excurNorm)**1; # TODO - 1/f
    
    # TODO - do we need to return H and weights
    return H, weights, sos, gain
            
def _costFunction(paramsEst:np.array,
                  filterType,
                  fVec,
                  excurNorm,
                  fs):
    
    H, weights, _, _ = _fitToSealed(paramsEst, filterType, fVec, excurNorm, fs)
    
    # calc error function
    err = (1 - (H / excurNorm)) * weights;
    sqSumErr = np.sum(np.abs(err));
    
    return sqSumErr
        
def designExcursionFilter(fVec,
                          excur,
                          wc,
                          excurGain,
                          excurMm,
                        #   HieqExcurOffset,
                          filterType,
                          enclosureType,
                          fs,
                          plotData:bool=False):
    
    # normalised excursion
    excurNorm = excur * wc**2

    # # removes inductance from excursion if required
    # excurNorm = excurNorm * HieqExcurOffset

    # normalised gain for normalised excursion to mm excursion
    norm2mmGain = (excurGain * 1000) / wc**2
    
    # optimisation of filter parameters
    match enclosureType:
        # fit against a sealed enclosure model
        case 'sealed':
            
            match filterType:
                case 'lp':
                    paramsEst = np.array([1, 70, 0.707]) # [ gain lpFc lpQ  ]
                case 'lppeq':
                    paramsEst = np.array([1, 70, 0.707, 40, 0.5,  1]) # [ gain lpFc lpQ peqFc peqGain peqQ ]
                case 'lp2peq':
                    paramsEst = np.array([1, 70, 0.707, 40, 0.5, 1, 100, -0.5, 1]) # [ gain lpFc lpQ peqFc1 peqGain1 peqQ1 peqFc2 peqGain2 peqQ2]
            
            # optimisation function to find fitted filter parameters
            args = (filterType, fVec, excurNorm, fs)
            fittedData = opt.fmin(_costFunction, paramsEst, args=args, xtol=1e-10, ftol=1e-6, maxiter=10e3, maxfun=10e3, full_output=True)
            
            # runs fitted params through function to get final sos matrix and gain value
            H, weights, sos, gain = _fitToSealed(fittedData[0], filterType, fVec, excurNorm, fs)
            
        case 'ported':
            # TODO - implement
            pass
        
        case _:
            raise ValueError('Error: invalid enclosureType')

    if plotData:
        
        H = excurGain * sig.sosfreqz(sos, fVec, fs=fs)
        
        # calculate overall transfer function
        H = gain * sig.sosfreqz(sos, fVec, fs)
    
        # calculate individual transfer functions of lp and peq filters used
        Hlp = sig.sosfreqz(sos[0,:], fVec, fs)
        Hpeq = gain * sig.sosfreqz(sos[1,:], fVec, fs)

        # plot data
        plt.figure()
        plt.subplot(2,2,1)
        plt.semilogx(fVec, np.abs(excurMm))
        plt.grid()
        plt.title('Displacement')
        plt.xlabel('frequency [Hz]')
        plt.ylabel('displacement [mm]')
        plt.xlim(fVec[0], fVec[-1])
        
        plt.subplot(2,2,2)
        plt.semilogx(fVec, np.abs(excurNorm), label='dispNorm')
        plt.semilogx(fVec, np.abs(H), label='dispNorm')
        plt.grid()
        plt.title('Original vs. Fitted Normalised Displacement')
        plt.xlabel('frequency [Hz]')
        plt.ylabel('displacement [normalised]')
        plt.xlim(fVec[0], fVec[-1])
        
        plt.subplot(2,2,3)
        plt.semilogx(fVec, np.abs(Hlp), label='Hlp')
        plt.semilogx(fVec, np.abs(Hpeq), label='Hpeq')
        plt.semilogx(fVec, np.abs(Hlp * Hpeq), label='Htot')
        plt.grid()
        plt.title('Fitted Displacement')
        plt.xlabel('frequency [Hz]')
        plt.ylabel('displacement [normalised]')
        plt.xlim(fVec[0], fVec[-1])
        
        plt.subplot(2,2,4)
        plt.semilogx(fVec, 20*np.log10(H / excurNorm), label='dispNorm')
        plt.grid()
        plt.title('Fitted Displacement Error')
        plt.xlabel('frequency [Hz]')
        plt.ylabel('magnitude [dB]')
        plt.xlim(fVec[0], fVec[-1])
    
# if __name__ == "__main__":
    
    
