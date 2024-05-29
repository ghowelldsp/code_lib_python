#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.signal._filter_design as fd

import codelibpython.dsp_maths as dspm

def _alignment(bS:np.array,
               aS:np.array,
               fs:int,
               fVec:np.array,
               plot:bool=False):
    
    # calculate discrete coefficients using bilinear transform
    bZ, aZ = sig.bilinear(bS, aS, fs)
    
    # determine pole zero locations
    z, p, k = sig.tf2zpk(bZ, aZ)
    
    def removeNyquistFilter(zp):
        
        # reorder in to complex conjugate pairs
        zp = fd._cplxpair(zp)
        
        # get the index relating to the Nyquist filter
        idx = np.where(np.real(zp) == zp)[0]
        if (idx.size == 0):
            raise ValueError('Error: no real components found')
        else:
            idx = idx[0]
        
        # nyquist filter and new alignment values
        zpNyq = np.array([zp[idx]])
        zpAlign = np.delete(zp, idx)
        
        return zpAlign, zpNyq
        
    # remove nyquist filter
    zAlign, zNyq = removeNyquistFilter(z)
    pAlign, pNyq = removeNyquistFilter(p)
    
    # check zero alignment parameters are within limits
    # TODO - check on the reasoning of this
    if (zNyq > -0.9) or (zNyq < -1.1):
        raise ValueError('Error: zeros over limits')
    
    # check for a 3rd order sytem
    if (len(zAlign) != 3) or (len(pAlign) != 3):
        raise ValueError('Error: not a 3rd order system')

    # calculate coefficient values of the 2nd order lowpass filter with normalised gain
    bAlignZ, aAlignZ = sig.zpk2tf(zAlign, pAlign, k=1)
    
    if plot:
        # create frequency vector and alignment transfer functions
        Hcont = sig.freqs(bS, aS, 2*np.pi*fVec)[1]
        Hdisc = sig.freqz(bZ, aZ, fVec, fs=fs)[1]
        Halign = sig.freqz(bAlignZ, aAlignZ, fVec, fs=fs)[1]
        bNyq, aNyq = sig.zpk2tf(zNyq, pNyq, k=1)
        HNyq = k * sig.freqz(bNyq, aNyq, fVec, fs=fs)[1]
        
        # plot data
        plt.figure()
        plt.semilogx(fVec, 20*np.log10(Hcont), label='continuous')
        plt.semilogx(fVec, 20*np.log10(Hdisc), label='discrete')
        plt.semilogx(fVec, 20*np.log10(Halign), label='alignment')
        plt.semilogx(fVec, 20*np.log10(HNyq), label='nyquist')
        plt.legend()
        plt.grid()
        plt.title('Alignment')
        plt.xlabel('freq [Hz]')
        plt.ylabel('magnitude [dB]')
        plt.xlim(fVec[0], fVec[-1])
        plt.ylim(-60, 10)
        
    return bAlignZ, aAlignZ

def _hpShelfFilters(bAlignZ:np.array,
                    aAlignZ:np.array,
                    fs:int,
                    fVec:np.array,
                    plot:bool=False):
    
    # transform 3rd order system into 2nd order sections (biquads)
    sos = sig.tf2sos(bAlignZ, aAlignZ)
    
    # TODO - the a coefficient are taken from different sections when the should be from the same row. Need to see why
    # the matlab implimentation is different
    
    # # get 1st order shelf
    # bShelf = sos[0,0:3]
    # aShelf = sos[1,3:6]

    # # get 2nd order highpass coefficients
    # bHp = sos[1,0:3]
    # aHp = sos[0,3:6]
    
    # get 1st order shelf
    bShelf = sos[0,0:3]
    aShelf = sos[0,3:6]

    # get 2nd order highpass coefficients
    bHp = sos[1,0:3]
    aHp = sos[1,3:6]
    
    # find the gain of the hp filter at the top end of the frequency spectrum where the gain is maximally flat
    freq = 0.9 * fs/2                                                        
    gain = np.abs(sig.freqz(bHp, aHp, freq, fs=fs)[1])
    bHpNorm = bHp / gain;
    bShelfNorm = bShelf * gain;

    if plot:
        
        # frequency vector and transfer function
        Halign = sig.freqz(bAlignZ, aAlignZ, fVec, fs=fs)[1]
        Hhp = sig.freqz(bHp, aHp, fVec, fs=fs)[1]
        Hshelf = sig.freqz(bShelf, aShelf, fVec, fs=fs)[1]
        HhpNorm = sig.freqz(bHpNorm, aHp, fVec, fs=fs)[1]
        HshelfNorm = sig.freqz(bShelfNorm, aShelf, fVec, fs=fs)[1]
        
        # plotting
        plt.figure()
        
        plt.subplot(2,1,1)
        plt.semilogx(fVec, 20*np.log10(Halign), '--', label='alignment')
        plt.semilogx(fVec, 20*np.log10(Hhp), label='highpass')
        plt.semilogx(fVec, 20*np.log10(Hshelf), label='shelf')
        plt.legend()
        plt.grid()
        plt.title('HP & Shelf TFs')
        plt.xlabel('freq [Hz]')
        plt.ylabel('magnitude [dB]')
        plt.xlim(fVec[0], fVec[-1])
        
        plt.subplot(2,1,2)
        plt.semilogx(fVec, 20*np.log10(Hhp), 'b--', label='highpass')
        plt.semilogx(fVec, 20*np.log10(Hshelf), 'g--', label='shelf')
        plt.semilogx(fVec, 20*np.log10(HhpNorm), 'b', label='highpass norm')
        plt.semilogx(fVec, 20*np.log10(HshelfNorm), 'g', label='shelf norm')
        plt.legend()
        plt.grid()
        plt.title('HP & Shelf Normalised TFs')
        plt.xlabel('freq [Hz]')
        plt.ylabel('magnitude [dB]')
        plt.xlim(fVec[0], fVec[-1])
        plt.ylim(-5, 5)
        
    return bShelfNorm, aShelf, bHp, aHp
    
def _createIeqFilter(bShelf:np.array,
                     aShelf:np.array,
                     fs:int,
                     fVec:np.array,
                     plot:bool=False):
    
    # invert shelf filter to form the ieq filter (inductance shelf compensation)
    bIeq = aShelf
    aIeq = bShelf

    # scale coefficients to make a0 = 1
    bIeq = bIeq / aIeq[0]; 
    aIeq = aIeq / aIeq[0];

    # check to see if the resulting ieq filter is stable
    # TODO - should this check for min phase too?
    if not dspm.isStable(aIeq):
        raise ValueError('Error: Ieq filter is unstable')
        
    if plot:
        # calculate frequency responses
        Hshelf = sig.freqz(bShelf, aShelf, fVec, fs=fs)[1]
        Hieq = sig.freqz(bIeq, aIeq, fVec, fs=fs)[1]
        
        # plot the 
        plt.figure()
        plt.semilogx(fVec, 20*np.log10(Hshelf), '--', label='shelf')
        plt.semilogx(fVec, 20*np.log10(Hieq), label='Ieq')
        plt.legend()
        plt.grid()
        plt.title('Ieq Filter')
        plt.xlabel('freq [Hz]')
        plt.ylabel('magnitude [dB]')
        plt.xlim(fVec[0], fVec[-1])
    
    return bIeq, aIeq

def _createXeqFilter(aHp:np.array,
                     ft:float,
                     Qt:float,
                     fs:int,
                     fVec:np.array,
                     plot:bool=False):
    
    # create 2nd order reference filter coefficients
    bHpRef, aHpRef = dspm.createFlt2ndOrderZ(ft, Qt, fs, filterType='highpass');

    # xeq (cancel original pole, and add extension pole, assume zeros the same)
    bXeq = aHp;
    aXeq = aHpRef;

    # check to see if the resulting xeq filter is stable
    # TODO - should this check for min phase too?
    if not dspm.isStable(aXeq):
        raise ValueError('Error: Xeq filter is unstable')
    
    if plot:
        # calculate frequency responses
        Hhp = sig.freqz(bHpRef, aHpRef, fVec, fs=fs)[1]
        Hxeq = sig.freqz(bXeq, aXeq, fVec, fs=fs)[1]
        
        # plot the 
        plt.figure()
        plt.semilogx(fVec, 20*np.log10(Hhp), '--', label='highpass')
        plt.semilogx(fVec, 20*np.log10(Hxeq), label='Xeq')
        plt.legend()
        plt.grid()
        plt.title('Ieq Filter')
        plt.xlabel('freq [Hz]')
        plt.ylabel('magnitude [dB]')
        plt.xlim(fVec[0], fVec[-1])
        
    return bXeq, aXeq

def designXeqIeqFilters(bAlignS:np.array,
                        aAlignS:np.array,
                        ft:float,
                        Qt:float,
                        fs:int,
                        plot:bool=False):
    
    if plot:
        # create frequency vector
        # TODO - create log vector, and move into parent class
        fVec = np.linspace(10, fs/2, 10000)
    else:
        fVec = None
    
    # create filters
    bAlignZ, aAlignZ = _alignment(bAlignS, aAlignS, fs, fVec, plot)
    bShelf, aShelf, bHp, aHp = _hpShelfFilters(bAlignZ, aAlignZ, fs, fVec, plot)
    bIeq, aIeq = _createIeqFilter(bShelf, aShelf, fs, fVec, plot)
    bXeq, aXeq = _createXeqFilter(aHp, ft, Qt, fs, fVec, plot)
    
    return bIeq, aIeq, bXeq, aXeq
