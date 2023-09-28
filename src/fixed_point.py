#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 19:06:48 2023

@author: ghowell
"""

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
    if (fVal > fMax):
        warnings.warn("float value %f saturated, max float value is %f" % (fVal, fMax))      
        fVal = fMax
    elif (fVal < fMin):
        warnings.warn("float value %f saturated, min float value is %f" % (fVal, fMin)) 
        fVal = fMin
    elif (fVal < fracVal and fVal > -fracVal and signed):
        warnings.warn("float value %f saturated, min float value is %f" % (fVal, fMin)) 
        fVal = 0
        
    # calculate fixed point value
    fpVal = int(fVal * 2**fracLen)
    
    return fpVal, fMin, fMax, fracVal

def fixedToFloat(fVal, fracLen):
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