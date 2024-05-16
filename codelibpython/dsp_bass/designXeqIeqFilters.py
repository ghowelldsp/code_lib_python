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

class designXeqIeqFilters():
    
    def __init__(self,
                 fs:int,
                 plot:bool=False):
        
        self.fs = fs
        self.plot = plot
        
        if plot:
            # create frequency vector
            self.fVec = np.linspace(10, fs/2, 1000)
   
    def __alignment(self,
                    bS:np.array,
                    aS:np.array):
        
        # calculate discrete coefficients using bilinear transform
        bZ, aZ = sig.bilinear(bS, aS, self.fs)
        
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
        bAlign, aAlign = sig.zpk2tf(zAlign, pAlign, k=1)
        
        if self.plot:
            # create frequency vector and alignment transfer functions
            Hcont = sig.freqs(bS, aS, 2*np.pi*self.fVec)[1]
            Hdisc = sig.freqz(bZ, aZ, self.fVec, fs=self.fs)[1]
            Halign = sig.freqz(bAlign, aAlign, self.fVec, fs=self.fs)[1]
            bNyq, aNyq = sig.zpk2tf(zNyq, pNyq, k=1)
            HNyq = k * sig.freqz(bNyq, aNyq, self.fVec, fs=self.fs)[1]
            
            # plot data
            plt.figure()
            plt.semilogx(self.fVec, 20*np.log10(Hcont), label='continuous')
            plt.semilogx(self.fVec, 20*np.log10(Hdisc), label='discrete')
            plt.semilogx(self.fVec, 20*np.log10(Halign), label='alignment')
            plt.semilogx(self.fVec, 20*np.log10(HNyq), label='nyquist')
            plt.legend()
            plt.grid()
            plt.title('Alignment')
            plt.xlabel('freq [Hz]')
            plt.ylabel('magnitude [dB]')
            plt.xlim(self.fVec[0], self.fVec[-1])
            plt.ylim(-60, 10)
            
        return bAlign, aAlign

    def __hpShelfFilters(self,
                         bAlign:np.array,
                         aAlign:np.array):
        
        # transform 3rd order system into 2nd order sections (biquads)
        sos = sig.tf2sos(bAlign, aAlign, 'down');

        # get 2nd order HP coefficients
        bHp = sos[0,0:3];
        aHp = sos[0,3:6];

        # get 1st order shelf
        bShelf = g * sos[1,0:3];
        aShelf = sos[1,3:6];
        
        # find the gain of the hp filter at the top end of the frequency spectrum where the gain is maximally flat
        # TODO - tidy this up
        # TODO - plot
        ftgt = 0.9 * self.fs/2;                                                         
        gainB2 = np.abs(sig.freqz(bHp, aHp, [ftgt, ftgt], fs=self.fs)); 
        gainB2 = gainB2[0];
        bHp = bHp / gainB2;
        bShelf = bShelf * gainB2;

        if self.plot:
            # frequency vector and transfer functionself.s
            Halign = sig.freqz(bAlign, aAlign, self.fVec, fs=self.fs)[1]
            Hhp = sig.freqz(bHp, aHp, self.fVec, fs=self.fs)[1]
            Hshelf = sig.freqz(bShelf, aShelf, self.fVec, fs=self.fs)[1]
            
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
        
    def __createIeqFilter(self,
                          bShelf:np.array,
                          aShelf:np.array):
        
        # ieq = inductance shelf compensation, invert shelf filter
        bIeq = aShelf
        aIeq = bShelf

        # scale coefficients to make a0 = 1
        bIeqNorm = bIeq / aIeq[0]; 
        aIeqNorm = aIeq / aIeq[0];

        # check to see if the resulting ieq filter is stable
        # TODO - should this check for min phase too?
        if ~dspm.isStable(aIeq, 1)[0]:
            raise ValueError('Error: Ieq filter is unstable')
            
        if self.plot:
            # calculate frequency responses
            Hshelf = sig.freqz(bShelf, aShelf, self.fVec, fs=self.fs)[1]
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
        [bHpRef, aHpRef] = dspm.filter2ndOrderZ(ft, Qt, self.fs, ft, 'HP', 28);

        # xeq = extension EQ compensation
        # Cancel original pole, and add extension pole. Assume zeros the same
        bXeq = aHp;
        aXeq = aHpRef;

        % check to see if the resulting xeq filter is stable
        if ~checkfilter(bXeq, aXeq, 1)
            error('XEQ filter is unstable')
    
    
# def _mainPlots():

    def process(self,
                tuningParams):
        
        self.__alignment()
        self.__hpShelfFilters()
        self.__createIeqFilter()
        self.__createXeqFilter()
        self.__mainPlots()
    