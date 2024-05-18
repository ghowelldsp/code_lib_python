#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.signal._filter_design as fd

import dsp_maths as dspm

def _alignment(bAlignS:np.array,
               aAlignS:np.array,
               fs:int,
               fVec:np.array):
    
    # calculate discrete coefficients using bilinear transform
    bZ, aZ = sig.bilinear(bAlignS, aAlignS, fs)
    
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
    if (zAlign > -0.9) or (zAlign > -1.1):
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

def __hpShelfFilters(self):
    
    # transform 3rd order system into 2nd order sections (biquads)
    sos = sig.tf2sos(self.bAlign, self.aAlign, 'down');

    # get 2nd order HP coefficients
    self.bHp = sos[0,0:3];
    self.aHp = sos[0,3:6];

    # get 1st order shelf
    self.bShelf = g * sos[1,0:3];
    self.aShelf = sos[1,3:6];
    
    # find the gain of the hp filter at the top end of the frequency spectrum where the gain is maximally flat
    # TODO - tidy this up
    # TODO - plot
    ftgt = 0.9 * self.fs/2;                                                         
    gainB2 = np.abs(sig.freqz(self.bHp, self.aHp, [ftgt, ftgt], fs=self.fs)); 
    gainB2 = gainB2[0];
    self.bHp = self.bHp / gainB2;
    self.bShelf = self.bShelf * gainB2;

    if self.plot:
        # frequency vector and transfer functionself.s
        Halign = sig.freqz(self.bAlign, self.aAlign, self.fVec, fs=self.fs)[1]
        Hhp = sig.freqz(self.bHp, self.aHp, self.fVec, fs=self.fs)[1]
        Hshelf = sig.freqz(self.bShelf, self.aShelf, self.fVec, fs=self.fs)[1]
        
        # plotting
        plt.figure()
        plt.semilogx(self.fVec, 20*np.log10(Halign), label='alignment')
        plt.semilogx(self.fVec, 20*np.log10(Hhp), label='highpass')
        plt.semilogx(self.fVec, 20*np.log10(Hshelf), label='shelf')
        plt.legend()
        plt.grid()
        plt.title('HP & Shelf TFs')
        plt.xlabel('freq [Hz]')
        plt.ylabel('magnitude [dB]')
        plt.xlim(self.fVec[0], self.fVec[-1])
        plt.ylim(-60, 10)
    
def __createIeqFilter(self):
    
    # ieq = inductance shelf compensation, invert shelf filter
    bIeq = self.aShelf
    aIeq = self.bShelf

    # scale coefficients to make a0 = 1
    bIeqNorm = bIeq / aIeq[0]; 
    aIeqNorm = aIeq / aIeq[0];

    # check to see if the resulting ieq filter is stable
    # TODO - should this check for min phase too?
    if ~dspm.isStable(aIeq, 1)[0]:
        raise ValueError('Error: Ieq filter is unstable')
        
    if self.plot:
        # calculate frequency responses
        Hshelf = sig.freqz(self.bShelf, self.aShelf, self.fVec, fs=self.fs)[1]
        Hieq = sig.freqz(bIeq, aIeq, self.fVec, fs=self.fs)[1]
        HieqNorm = sig.freqz(bIeqNorm, aIeqNorm, self.fVec, fs=self.fs)[1]
        
        # plot the 
        plt.figure()
        plt.semilogx(self.fVec, 20*np.log10(Hshelf), label='shelf')
        plt.semilogx(self.fVec, 20*np.log10(Hieq), label='Ieq')
        plt.semilogx(self.fVec, 20*np.log10(HieqNorm), label='Ieq Norm')
        plt.legend()
        plt.grid()
        plt.title('Ieq Filter')
        plt.xlabel('freq [Hz]')
        plt.ylabel('magnitude [dB]')
        plt.xlim(self.fVec[0], self.fVec[-1])
        plt.ylim(-60, 10)

def __createXeqFilter(self,
                        ft,
                        Qt):
    
    # create 2nd order reference filter coefficients
    bHpRef, aHpRef = dspm.createFlt2ndOrderZ(ft, Qt, self.fs, filterType='highpass');

    # xeq (cancel original pole, and add extension pole, assume zeros the same)
    bXeq = aHp;
    aXeq = aHpRef;

    # check to see if the resulting xeq filter is stable
    # TODO - should this check for min phase too?
    if ~dspm.isStable(aXeq, 1)[0]:
        raise ValueError('Error: Xeq filter is unstable')

# def __mainPlots():

def designXeqIeqFilters(bAlignS:np.array,
                        aAlignS:np.array,
                        ft:float,
                        Qt:float,
                        fs:int,
                        plot:bool=False):
    
    if plot:
        # create frequency vector
        self.fVec = np.linspace(10, fs/2, 1000)
    
    bAlignZ, aAlignZ = self._alignment(bAlignS, aAlignS)
    # self.__hpShelfFilters()
    # self.__createIeqFilter()
    # self.__createXeqFilter(ft, Qt)
    # self.__mainPlots()
