#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

import numpy as np
import scipy.signal as sig

def tf2sos(b,
           a):
    
    # fractorise to zero / poles
    z, p, k = sig.tf2zpk(b, a)
    
    # reorder the poles so that the ones closest to the unit circle are first
    # TODO - should this be done to zeros too
    sortIdx = np.argsort(np.abs(np.abs(p) - 1))
    print(sortIdx)
    p = p[sortIdx]
    z = z[sortIdx]
    
    sos = sig.zpk2sos(z, p, k, pairing='nearest')
    
    return sos
    
if __name__ == "__main__":
    
    b = np.array([1, -2.9026596363, 2.8053192726, -0.9026596363])
    a = np.array([1, -2.9545395653, 2.910025178, -0.9554800283])
    
    sos =   (b, a)

    print(sos)
