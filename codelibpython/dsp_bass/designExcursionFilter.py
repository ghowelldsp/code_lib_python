#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

import numpy as np
import scipy.signal as sig
import scipy.optimize as opt
import dsp_maths as dspm

def _fitToSealed(paramsEst:np.array,
                 paramsFix:dict):
    
    # load variables
    fs = paramsFix['fs']
    fVec = paramsFix['fVec']
    excurNorm = paramsFix['excurNorm']
    
    match paramsFix['filterType']:
        
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
            sos = np.array([lpB, lpA])
            
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
            sos = np.array([[lpB, lpA], [peqB, peqA]])
        
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
            
    # calc transfer function of filters
    H = gain * sig.sosfreqz(sos, fVec, fs)[1]

    # calculate weights                          
    weights = np.abs(excurNorm)**1; # TODO - 1/f
    
    return H, weights
            
def _costFunction(paramsEst:np.array,
                  paramsFix:dict):
    
    H, weights = _fitToSealed(paramsEst, paramsFix)
    
    # calc error function
    err = (1 - (H / paramsFix['excurNorm'])) * weights;
    sqSumErr = np.sum(np.abs(err));
    
    return sqSumErr
        
def designExcursionFilter():
    
    # TODO - get inputs
    fVec 
    excur
    wc
    excurGain
    excurMm
    HieqExcurOffset
    filterType
    enclosureType 
    
    # normalised excursion
    excurNorm = excur * wc**2

    # removes inductance from excursion if required
    excurNorm = excurNorm * HieqExcurOffset                                

    # normalised gain for normalised excursion to mm excursion
    norm2mmGain = (excurGain * 1000) / wc**2
    
    # optimisation of filter parameters
    match enclosureType:
        # fit against a sealed enclosure model
        case 'sealedBox':
            
            match filterType:
                case 'lp':
                    paramsEst = np.array([1, 70, 0.707]) # [ gain lpFc lpQ  ]
                case 'lppeq':
                    paramsEst = np.array([1, 70, 0.707, 40, 0.5,  1]) # [ gain lpFc lpQ peqFc peqGain peqQ ]
                case 'lp2peq':
                    paramsEst = np.array([1, 70, 0.707, 40, 0.5, 1, 100, -0.5, 1]) # [ gain lpFc lpQ peqFc1 peqGain1 peqQ1 peqFc2 peqGain2 peqQ2]
            
            paramsFix = {
                'filterType' : filterType,
                'fs' : fs,
                'fVec' : freqArr,
                'excurNorm' : excurNorm
            }
            
            # optimisation function to find fitted filter parameters
            fittedData = opt.fmin(_costFunction, paramsEst, paramsFix, xtol=1e-10, ftol=1e-6, maxiter=10e3, maxfun=10e3, full_output=True)
            
            # runs fitted params through function to get final sos matrix and gain value
            H = _fitToSealed(fittedData[0], paramsFix)[0]
            
        case 'ported':
            # TODO - implement
            pass
        
        case _:
            raise ValueError('Error: invalid enclosureType')
        
    
    # TODO - plot data
    
if __name__ == "__main__":
    
    
