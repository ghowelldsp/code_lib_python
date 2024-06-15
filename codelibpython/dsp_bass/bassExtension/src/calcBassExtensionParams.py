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
import codelibpython.dsp_bass.bassExtension.src as src

def _calcAlignmentFlt(alignB:np.array,
                      alignA:np.array,
                      fs:int,
                      fVec:np.array,
                      plot:bool=False):
    """ Calculate Alignment Filter

    Parameters
    ----------
    alignB : np.array
        Alignment b coefficients in the continuous domain. TODO - check form
    alignA : np.array
        Alignment a coefficients in the continuous domain. TODO - check form
    fs : int
        Samples rate [Hz].
    fVec : np.array
        Frequency vector.
    plot : bool, optional
        Plot data. Defaults to False.

    Returns
    -------
    # TODO
    """
    
    # calculate discrete coefficients using bilinear transform
    bZ, aZ = sig.bilinear(alignB, alignA, fs)
    
    # determine pole zero locations
    z, p, k = sig.tf2zpk(bZ, aZ)
    
    # polar plot 
    # TODO
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(np.angle(z), np.abs(z), 'o')
    ax.plot(np.angle(p), np.abs(p), '*')
    plt.show()
    
    def removeNyquistFilter(zp):
        
        # reorder in to complex conjugate pairs
        zp = fd._cplxpair(zp)
        
        # get the index relating to the Nyquist filter
        idx = np.where(np.real(zp) == zp)[0]
        if (idx.size == 0):
            raise ValueError('Error: no real components found')
        else:
            # idx = np.where(np.min(zp[idx]) == zp)[0]
            idx = idx[0]
        
        # nyquist filter and new alignment values
        zpNyq = np.array([zp[idx]])
        zpAlign = np.delete(zp, idx)
        
        return zpAlign, zpNyq
        
    # remove nyquist filter
    zAlign, zNyq = removeNyquistFilter(z)
    pAlign, pNyq = removeNyquistFilter(p)
    # temp
    # pAlign[2] = p[2]
    # pNyq[0] = p[3]
    
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
        # # create frequency vector and alignment transfer functions
        # Hcont = sig.freqs(alignB, alignA, 2*np.pi*fVec)[1]
        # Hdisc = sig.freqz(bZ, aZ, fVec, fs=fs)[1]
        # Halign = sig.freqz(bAlignZ, aAlignZ, fVec, fs=fs)[1]
        # bNyq, aNyq = sig.zpk2tf(zNyq, pNyq, k=1)
        # HNyq = k * sig.freqz(bNyq, aNyq, fVec, fs=fs)[1]
        
        # polar plot 
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(np.angle(z), np.abs(z), 'o')
        ax.plot(np.angle(p), np.abs(p), '*')
        plt.show()
        # ax.set_rmax(2)
        # ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
        # ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        # ax.grid(True)
        # ax.set_title("A line plot on a polar axis", va='bottom')
        
        # # plot data
        # plt.figure()
        # plt.semilogx(fVec, dspm.linToDb(Hcont), label='continuous')
        # plt.semilogx(fVec, dspm.linToDb(Hdisc), label='discrete')
        # plt.semilogx(fVec, dspm.linToDb(Halign), label='alignment')
        # plt.semilogx(fVec, dspm.linToDb(HNyq), label='nyquist')
        # plt.legend()
        # plt.grid()
        # plt.title('Alignment')
        # plt.xlabel('freq [Hz]')
        # plt.ylabel('magnitude [dB]')
        # plt.xlim(fVec[0], fVec[-1])
        # plt.ylim(-60, 10)
        
    return bAlignZ, aAlignZ

def _calcHpShelfFlt(bAlignZ:np.array,
                    aAlignZ:np.array,
                    fs:float,
                    fVec:np.array,
                    plot:bool=False):
    """ Calculate Highpass Shelf Filter

    Parameters
    ----------
    alignB : np.array
        Alignment b coefficients in the discrete domain. TODO - check form
    alignA : np.array
        Alignment a coefficients in the discrete domain. TODO - check form
    fs : int
        Samples rate [Hz].
    fVec : np.array
        Frequency vector.
    plot : bool, optional
        Plot data. Defaults to False.

    Returns:
    TODO
    """
    
    # transform 3rd order system into 2nd order sections (biquads)
    sos = sig.tf2sos(bAlignZ, aAlignZ)
    
    # TODO - update
    # the a coefficient are taken from different sections when the should be from the same row. Need to see why
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
        # TODO - convert alll logs to linToDb
        plt.semilogx(fVec, dspm.linToDb(Halign), '--', label='alignment')
        plt.semilogx(fVec, dspm.linToDb(Hhp), label='highpass')
        plt.semilogx(fVec, dspm.linToDb(Hshelf), label='shelf')
        plt.legend()
        plt.grid()
        plt.title('HP & Shelf TFs')
        plt.xlabel('freq [Hz]')
        plt.ylabel('magnitude [dB]')
        plt.xlim(fVec[0], fVec[-1])
        
        plt.subplot(2,1,2)
        plt.semilogx(fVec, dspm.linToDb(Hhp), 'b--', label='highpass')
        plt.semilogx(fVec, dspm.linToDb(Hshelf), 'g--', label='shelf')
        plt.semilogx(fVec, dspm.linToDb(HhpNorm), 'b', label='highpass norm')
        plt.semilogx(fVec, dspm.linToDb(HshelfNorm), 'g', label='shelf norm')
        plt.legend()
        plt.grid()
        plt.title('HP & Shelf Normalised TFs')
        plt.xlabel('freq [Hz]')
        plt.ylabel('magnitude [dB]')
        plt.xlim(fVec[0], fVec[-1])
        plt.ylim(-5, 5)
        
    return bShelfNorm, aShelf, bHp, aHp

def _createInductanceFlt(bShelf:np.array,
                         aShelf:np.array,
                         fs:int,
                         fVec:np.array,
                         plot:bool=False):
    
    # invert shelf filter to form the ieq filter (inductance shelf compensation)
    bInd = aShelf
    aInd = bShelf

    # scale coefficients to make a0 = 1
    bInd = bInd / aInd[0]; 
    aInd = aInd / aInd[0];

    # check to see if the resulting ieq filter is stable
    # TODO - should this check for min phase too?
    if not dspm.isStable(aInd):
        raise ValueError('Error: Ieq filter is unstable')
        
    if plot:
        # calculate frequency responses
        Hshelf = sig.freqz(bShelf, aShelf, fVec, fs=fs)[1]
        Hind = sig.freqz(bInd, aInd, fVec, fs=fs)[1]
        
        # plot the 
        plt.figure()
        plt.semilogx(fVec, dspm.linToDb(Hshelf), '--', label='shelf')
        plt.semilogx(fVec, dspm.linToDb(Hind), label='inductance')
        plt.legend()
        plt.grid()
        plt.title('Inductance Filter')
        plt.xlabel('freq [Hz]')
        plt.ylabel('magnitude [dB]')
        plt.xlim(fVec[0], fVec[-1])
    
    return bInd, aInd

def _createExtensionFlt(aHp:np.array,
                        fcExt:float,
                        qExt:float,
                        fs:int,
                        fVec:np.array,
                        plot:bool=False):
    
    # create 2nd order reference filter coefficients
    bHpRef, aHpRef = dspm.createFlt2ndOrderZ(fcExt, qExt, fs, filterType='highpass');

    # xeq (cancel original pole, and add extension pole, assume zeros the same)
    bExt = aHp;
    aExt = aHpRef;

    # check to see if the resulting xeq filter is stable
    # TODO - should this check for min phase too?
    if not dspm.isStable(aExt):
        raise ValueError('Error: Xeq filter is unstable')
    
    if plot:
        # calculate frequency responses
        Hhp = sig.freqz(bHpRef, aHpRef, fVec, fs=fs)[1]
        Hxeq = sig.freqz(bExt, aExt, fVec, fs=fs)[1]
        
        # plot the 
        plt.figure()
        plt.semilogx(fVec, dspm.linToDb(Hhp), '--', label='highpass')
        plt.semilogx(fVec, dspm.linToDb(Hxeq), label='extension')
        plt.legend()
        plt.grid()
        plt.title('Extension Filter')
        plt.xlabel('freq [Hz]')
        plt.ylabel('magnitude [dB]')
        plt.xlim(fVec[0], fVec[-1])
        
    return bExt, aExt

def _calcMaxRmsXeqFilter(fVec:np.array,
                         sos:np.array,
                         gain:float,
                         gainToMm:float,
                         bIeq:np.array,
                         aIeq:np.array,
                         aHp:np.array,
                         fcLowExt:float,
                         qExt:float,
                         voltsPeakAmp:float,
                         maxDispLimit:float,
                         maxVoltLimit:float,
                         fs:float,
                         plotData:bool=True):
    """ Calculate Max RMS Extension Filter
    
    Typically, the desired cutoff frequency of the speaker response after the bass extention filter is applied 
    results in a displacement and voltage level that is over required limits when a maximum RMS signal level is
    input. In order to then determine the appropriate extension filter that should be applied to avoid this, the signal
    cutoff frequency is iterativly increased until both these limits are met. 

    Parameters
    ----------
    fVec : np.array
        1D frequency vector
    dispSos : np.array [stages][coefficients]
        Displace filter sos coefficients. In the form [b0, b1, b2, a0, a1, a2].
    dispGain : float
        Displacement gain.
    gainToMm : float
        Displacement gain to in mm.
    bInd : np.array
        1D array of inductance b coefficients. In the form [b0, b1, b2].
    aInd : np.array
        1D array of inductance a coefficients. In the form [a0, a1, a2].
    aHp : np.array
        # TODO
    fcLowExt : float
        Lowest cutoff frequency of the desired extension filter.
    qExt : float
        Quality (Q) value of desired extension filter.
    voltsPeakAmp : float
        Voltage level a maximum gain.
    maxDispLimit : float
        Maximum displacement limit. # TODO - find out if this is mm
    maxVoltLimit : float
        Maximum voltage limit.
    fs : int
        Samples rate [Hz].
    plot : bool, optional
        Plot data. Defaults to False.

    Returns
    -------
    # TODO
    """
    
    # limit frequency array
    fVecLimIdx = (fVec > 10) & (fVec < 200)
    
    # displacement and ieq filters
    HdispMmTmp = gainToMm * gain * sig.sosfreqz(sos, fVec, fs=fs)[1]
    Hieq = sig.freqz(bIeq, aIeq, fVec, fs=fs)[1]
    
    # initialise variables
    maxDisp = np.inf
    maxVolts = np.inf
    fcHigh = fcLowExt
    
    # calculate displacement and voltage for increasing cutoff frequencies untill limits have been met
    while not((maxDisp <= maxDispLimit) and (maxVolts <= maxVoltLimit)):
        
        # iteratively increase cutoff of desired output repsonse
        fcHigh = fcHigh + 0.5;
        
        # calculate extension filter
        bExtHigh, aExtHigh = _createExtensionFlt(aHp, fcHigh, qExt, fs, fVec)           
        HextHigh = sig.freqz(bExtHigh, aExtHigh, fVec, fs=fs)[1]
        
        # # determine TF of HP reference filter (for plotting)
        # TODO - used for the animation plot
        # bHpRef, aHpRef = dspm.createFlt2ndOrderZ(ftHigh, Qt, self.fs, warp=True, filterType='highpass')
        # HhpRef = sig.freqz(bHpRef, aHpRef, fVec, fs=self.fs)[1]
        
        # calculate maximum displacement in mm
        HdispMm = HextHigh * Hieq * HdispMmTmp
        maxDisp = np.max(np.abs(HdispMm[fVecLimIdx]))
        
        # calculate maximum voltage
        Hvolts = HextHigh * Hieq * voltsPeakAmp
        maxVolts = np.max(np.abs(Hvolts[fVecLimIdx]))
        
    # plotting
    if plotData:
        
        # TODO - calculate
        
        fig, ax1 = plt.subplots()
        
        plt.title(f'Displacement and Voltage for Max RMS Input - Fc = {fcHigh} Hz')

        color = 'tab:blue'
        ax1.semilogx(fVec, np.abs(HdispMm), color=color, label='_nolegend_')
        lns1 = ax1.semilogx(np.array([fVec[0], fVec[-1]]), np.array([maxDispLimit, maxDispLimit]), '--', color=color, 
                        label='Max Displacement Limit')
        ax1.grid()
        ax1.set_xlabel('frequency [Hz]')
        ax1.set_ylabel('Displacement [mm]', color=color)
        ax1.set_xlim(fVec[0], 1000)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

        color = 'tab:red'
        ax2.plot(fVec, np.abs(Hvolts), color=color, label='_nolegend_')
        lns2 = ax2.semilogx(np.array([fVec[0], fVec[-1]]), np.array([maxVoltLimit, maxVoltLimit]), '--', color=color,
                        label='Max Voltage Limit')
        ax2.set_ylabel('Volts [V]', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # added these three lines
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=6)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    return bExtHigh, aExtHigh, fcHigh
    
def _rmsThreshold(bExtLo,
                  aExtLo,
                  bExtHi,
                  aExtHi,
                  bInd,
                  aInd,
                  Hdisp,
                  voltsPeakAmp,
                  maxDispLimit,
                  maxVoltLimit,
                  exc2mmGain,
                  fVec,
                  fs,
                  plot):
    
    # create extension and inductance filterss
    HextLoRms = sig.freqz(bExtLo, aExtLo, fVec, fs=fs)[1]
    HextHiRms = sig.freqz(bExtHi, aExtHi, fVec, fs=fs)[1]
    Hind = sig.freqz(bInd, aInd, fVec, fs=fs)[1]
    
    # finds 2 index values of the frequency vector that are closest to 20Hz 
    # TODO - this was originally 2 points, not sure why
    fVecIdx = np.where(fVec >= 20)[0][0]
    
    # array of amplitude values
    nAmps = 2000
    amps = np.linspace(0, 1.2, nAmps)

    # loop through amplitudes until A needs to start proctecting the output (that will be the threshold)
    
    # determines the max mm and max voltage at the various amplitude levels, stops once the max mm or volts level is 
    # reached with the resulting amplitude being used as the threshold
    for i in range(nAmps-1):
        
        # calculate max mm and volts
        maxDisp = np.max(np.abs(HextLoRms[fVecIdx] * Hind[fVecIdx] * Hdisp[fVecIdx] *  exc2mmGain)) * amps[i+1]
        maxVolts = np.max(np.abs(HextLoRms[fVecIdx] * Hind[fVecIdx] * voltsPeakAmp)) * amps[i+1]
        
        # check mm and volt limits
        if ((maxDisp > maxDispLimit) or (maxVolts > maxVoltLimit)):
            threshold = amps[i-1]
            break

    # determine gradient and y intercept
    grad = (1 - 0)/(1 - threshold)
    yInt = 1 - grad
    
    # figure out gain of an equivalent HPF with the same poles as poleExt
    poleExtLow = np.roots(aExtLo)[0]
    poleExtHigh = np.roots(aExtHi)[0]
    kLow = np.abs(poleExtLow)
    kLowInv = 1 / kLow
    
    if plot:
        
        # calculate output
        y = grad * amps + yInt

        plt.figure()
        plt.plot(dspm.linToDb(amps), y)
        plt.grid()
        plt.xlabel('RMS Amplitude [dB FS]')
        plt.ylabel('Low (0) -> High (1) RMS DEQ Filter')
        plt.title('RMS Ampitude / Filter Type Determination')
        plt.ylim(0, 1)
        
    return grad, yInt, poleExtLow, poleExtHigh, kLow, kLowInv

def calcBassExtensionParams(driverParams:dict,
                            fcLowExt:float,
                            qExt:float,
                            maxMmPeak:float,
                            maxVoltPeak:float,
                            smoothAttackTime:float,
                            smoothReleaseTime:float,
                            rmsAttackTime:float,
                            fs:float,
                            dropInd:bool=False,
                            plot:bool=False):


    # TODO - create log vector
    fVec = np.linspace(10, fs/2, 10000)
    
    # calculate alignment
    bAlignZ, aAlignZ = _calcAlignmentFlt(driverParams['bAlign'], driverParams['aAlign'], fs, fVec, plot=plot)
    
    # calculate highpass shelf filter (used to create inductance filter)
    bShelf, aShelf, bHp, aHp = _calcHpShelfFlt(bAlignZ, aAlignZ, fs, fVec, plot=plot)
    
    # create inducatance filter
    bInd, aInd = _createInductanceFlt(bShelf, aShelf, fs, fVec, plot=plot)
    
    # create extension filter
    bExtLow, aExtLow = _createExtensionFlt(aHp, fcLowExt, qExt, fs, fVec, plot=plot)
    
    # TODO - impliment drop inductance filter
    
    # calculate displacement filter
    sosDisp, gain, norm2mmGain, Hdisp = src.calcDisplacementFilter(driverParams['fVec'], driverParams['Hdisp'], driverParams['w0'],
                                                             driverParams['dispGain'], driverParams['HdispMm'], filterType='lppeq',
                                                             enclosureType='sealed', fs=fs, plot=plot)
    
    # calculate excursion filter when for a maximum rms input level
    # TODO - tidy
    bExtHigh, aExtHigh, fcHigh = _calcMaxRmsXeqFilter(
        fVec, sosDisp, gain, norm2mmGain, bInd, aInd, aHp, fcLowExt, qExt, driverParams['voltsPeakAmp'],
                         maxMmPeak, maxVoltPeak, fs, plotData=plot)
    
    # calculate rms threshold level
    rmsGrad, rmsYInter, poleExtLow, poleExtHigh, kLow, kLowInv = _rmsThreshold(bExtLow, aExtLow, bExtHigh, aExtHigh, bInd, aInd, Hdisp, driverParams['voltsPeakAmp'], maxMmPeak, maxVoltPeak, 
                  norm2mmGain, fVec, fs, plot=plot)
    
    # calculate attack release coefficients    
    rmsAttackCoeff = dspm.timeToCoeffConst(rmsAttackTime, fs)
    smoothAttackCoeff = dspm.timeToCoeffConst(smoothAttackTime, fs)
    smoothReleaseCoeff = dspm.timeToCoeffConst(smoothReleaseTime, fs)
    
    # create sos matricies
    # TODO - should this happen on the function output
    sosExtLow = np.array(np.concatenate([bExtLow, aExtLow]), ndmin=2)
    sosExtHigh = np.array(np.concatenate([bExtHigh, aExtHigh]), ndmin=2)
    sosInd = np.array(np.concatenate([bInd, aInd]), ndmin=2)
                
    # create all data
    bassExtensionParams = {
        'rmsAttackCoeff' : rmsAttackCoeff,
        'rmsGain' : np.sqrt(2.0),
        'rmsGrad' : rmsGrad,
        'rmsYInter' : rmsYInter,
        'smoothAttackCoeff' : smoothAttackCoeff,
        'smoothReleaseCoeff' : smoothReleaseCoeff,
        'poleHigh' : poleExtHigh,
        'poleLow' : poleExtLow,
        'sosExtLow' : sosExtLow,
        'sosExtHigh' : sosExtHigh,
        'sosInd' : sosInd,
        'fs' : fs,
        'kLow' : kLow,
        'kLowInv' : kLowInv
    }            
    
    if plot:
        plt.show()
        
    return bassExtensionParams
    
if __name__ == "__main__":
    
    print('\nCalculating Bass Extension Parameters\n')
    
    # general parameters
    fs = 48000
    
    # tuning parameters
    fcLowExt = 40
    qExt = 0.65
    maxMmPeak = 1.4
    maxVoltPeak = 20
    attackTime = 0.001
    releaseTime = 0.100
    rmsAttackTime = 0.005
    dropInd = False
    
    # TODO - temp for testing form matlab file
    from pymatreader import read_mat
    data = read_mat('../impedTestData_old/01_ALB_IMP_DEQ_reformatted_lp.mat')['impDataLumpParams']
    
    # TODO - check if all these params are needed
    driverParams = {
        'fVec' : data['deqParams']['alignment']['freq'],
        'w0' : data['deqParams']['enclosure']['wc'],
        'bAlign' : data['deqParams']['alignment']['num2'],
        'aAlign' : data['deqParams']['alignment']['den2'],
        'Hdisp' : data['deqParams']['alignment']['excursion'],
        'dispGain' : data['fitImpData']['excurGain'],
        'HdispMm' : data['fitImpData']['excurMm'],
        'voltsPeakAmp' : data['klipParams']['VoltsPeakAmp']
    }
    
    # run model
    calcBassExtensionParams(driverParams, fcLowExt, qExt, maxMmPeak, maxVoltPeak, attackTime, releaseTime, 
                            rmsAttackTime, fs, plot=True)
    
    plt.show()
    
    print('\nFinished\n')
