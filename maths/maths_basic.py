#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ghowell
"""

import numpy as np

def db(linVal):
    """
    converts linear values to decibels values
    
    :param linVal: linear value
    
    :return dbVal: decibel value
    """ 

    dbVal = 20*np.log10(abs(linVal))

    return dbVal

if __name__ == "__main__":

    """
    db
    """
    print(db(0.5))
