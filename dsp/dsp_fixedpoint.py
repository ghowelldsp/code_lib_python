#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 19:06:48 2023

@author: ghowell
"""

import numpy as np
import warnings

def floatToFixed(fVal, fracLen, wordLen, signed):
    """
    converts a floating point number to a fixed point number
    
    :param fVal: floating point value
    :param fracBits: number of fractional bits
    :param wordLen: number of bits in the word
    
    :return fpVal: fixed point value
    :return fMin: floating point minimum value
    :return fMax: floating point maximum value
    """ 
    
    # number of integer bits
    intLen = wordLen - fracLen
    
    # float min / max value
    fMax = 0
    if signed:
        for i in range(wordLen):
            fMax += 2**(intLen-(2+i))
            fMin = -2**(intLen-1)
    else:
        for i in range(wordLen-1):
            fMax += 2**(intLen-(1+i))
            fMin = 2**(-fracLen)
    
    # fraction value
    fracVal = 2**(-fracLen)

    # saturate
    fVal = np.array(fVal)
    if (fVal > fMax).any():
        warnings.warn("positive overflow occured, max float value is %f" % fMax)
        fVal[np.where(fVal > fMax)] = fMax
    elif (fVal < fMin).any():
        warnings.warn("negative overflow occured, min float value is %f" % fMin)
        fVal[np.where(fVal < fMin)] = fMin
    elif ((fVal < np.abs(fracVal)).any() and signed):
        warnings.warn("underflow occured, min fractional float value is %f" % fracVal)
        
    # calculate fixed point value
    fpVal = fVal * 2**fracLen
    fpVal = np.array(fpVal, dtype=int)
    
    return fpVal, fMin, fMax, fracVal

def fixedToFloat(fpVal, fracLen):
    """
    converts a fixed point number to a floating point number
    
    :param fVal: floating point value
    :param fracBits: number of fractional bits
    :param wordLen: number of bits in the word
    
    :return fpVal: fixed point value
    :return fMin: floating point minimum value
    :return fMax: floating point maximum value
    """ 
    
    fVal = fpVal / 2**fracLen
    
    return fVal

if __name__ == "__main__":
    
    # calculate float as 2.2 fixed point number
    fpVal, fpMin, fpMax, fracVal = floatToFixed(3, 2, 4, False)
    
    print("fpVal = %d; fpMin = %f; fpMax = %f; fracVal = %f" % (fpVal, fpMin, fpMax, fracVal))
    
    # recalculate float value
    fVal = fixedToFloat(fpVal, 2)
    
    print("fVal = %f" % (fVal))