#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

import numpy as np

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
            lpB, lpA = createFlt2ndOrderZ(lpFc, lpQ, fs, filterType='lowpass')
            
            # form sos matrix
            sos = np.array([lpB, lpA])
        
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
            # fitFltParams = fminsearch(@(initFltParams) sys2fit_sealed(initFltParams, fixedparams), initFltParams);
            fittedData = opt.fmin(_fitToSealed, paramsEst, paramsFixed xtol=1e-10, ftol=1e-6, maxiter=10e3, maxfun=10e3, full_output=True)
            
            # runs fitted params through function to get final sos matrix and  gain value
            # fixedparams.showplots = 0
            _fitToSealed(paramsEst, paramsFix)
            # [~, sos, gain] = sys2fit_sealed(fitFltParams, fixedparams);
            
        case 'ported':
            # TODO - implement
            pass
        
        case _:
            raise ValueError('Error: invalid enclosureType')
    
if __name__ == "__main__":
    
    
