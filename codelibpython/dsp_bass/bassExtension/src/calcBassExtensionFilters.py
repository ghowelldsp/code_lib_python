#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.signal._filter_design as fd

import calcExcursionFilter as cef
import codelibpython.dsp_bass.bassExtension.src.calcExcursionFilter as defi
import codelibpython.dsp_maths as dspm

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
    TODO
    """
    
    # calculate discrete coefficients using bilinear transform
    bZ, aZ = sig.bilinear(alignB, alignA, fs)
    
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
        Hcont = sig.freqs(alignB, alignA, 2*np.pi*fVec)[1]
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

def _calcHpShelfFlt(bAlignZ:np.array,
                    aAlignZ:np.array,
                    fs:int,
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

def _createInductanceFlt(bShelf:np.array,
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

def _createExtensionFlt(aHp:np.array,
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

def _calcMaxRmsXeqFilter(self,
                         fVec:np.array,
                         sos:np.array,
                         gain:float,
                         gainToMm:float,
                         bIeq:np.array,
                         aIeq:np.array,
                         extFcLow:float,
                         extQ:float,
                         voltsPeakAmp:float,
                         maxDispLimit:float,
                         maxVoltLimit:float,
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
    extFcLow : float
        Lowest cutoff frequency of the desired extension filter.
    extQ : float
        Quality (Q) value of desired extension filter.
    voltsPeakAmp : float
        Voltage level a maximum gain.
    maxDispLimit : float
        Maximum displacement limit. # TODO - find out if this is mm
    maxVoltLimit : float
        Maximum voltage limit.
    plot : bool, optional
        Plot data. Defaults to False.

    Returns
    -------
    # TODO
    """
    
    # limit frequency array
    fVecLimIdx = (fVec > 10) & (fVec < 200)
    
    # displacement and ieq filters
    HdispMmTmp = gainToMm * gain * sig.sosfreqz(sos, fVec, fs=self.fs)[1]
    Hieq = sig.freqz(bIeq, aIeq, fVec, fs=self.fs)[1]
    
    # initialise variables
    maxDisp = np.inf
    maxVolts = np.inf
    fcHigh = extFcLow
    
    # calculate displacement and voltage for increasing cutoff frequencies untill limits have been met
    while not((maxDisp <= maxDispLimit) and (maxVolts <= maxVoltLimit)):
        
        # iteratively increase cutoff of desired output repsonse
        fcHigh = fcHigh + 0.5;
        
        # calculate extension filter
        _, _, bXeqHigh, aXeqHigh = dxif.designXeqIeqFilters(self.driverParams['bAlign'], 
                                                            self.driverParams['aAlign'], 
                                                            fcHigh, extQ, self.fs)           
        HxeqHigh = sig.freqz(bXeqHigh, aXeqHigh, fVec, fs=self.fs)[1]
        
        # # determine TF of HP reference filter (for plotting)
        # TODO - used for the animation plot
        # bHpRef, aHpRef = dspm.createFlt2ndOrderZ(ftHigh, Qt, self.fs, warp=True, filterType='highpass')
        # HhpRef = sig.freqz(bHpRef, aHpRef, fVec, fs=self.fs)[1]
        
        # calculate maximum displacement in mm
        HdispMm = HxeqHigh * Hieq * HdispMmTmp
        maxDisp = np.max(np.abs(HdispMm[fVecLimIdx]))
        
        # calculate maximum voltage
        Hvolts = HxeqHigh * Hieq * voltsPeakAmp
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
    
    return fcHigh
    
def _rmsThreshold():
    
    # IEQ and XEQ low rms fitler tf's
    [H_xeqLoRms] = freqz(bXeqLo, aXeqLo, freqArr, obj.fs)
    [H_xeqHiRms] = freqz(bXeqHI, aXeqHI, freqArr, obj.fs)
    [H_ieq] = freqz(bIeq, aIeq, freqArr, obj.fs)
    
    # finds 2 index values of the frequency vector that are closest to 20Hz 
    idx = find(freqArr >= 20, 2)
    
    # array of amplitude values
    amps = linspace(0, 1.2, 2000)'

    # loop through amplitudes until A needs to start proctecting the output (that will be the threshold)
    
    # determines the max mm and max voltage at the various amplitude levels, stops once the max mm or volts level is 
    # reached with the resulting amplitude being used as the threshold
    for n=2:length(amps)
        
        % calculate max mm and volts
        mm = max(abs(H_xeqLoRms(idx) .* H_ieq(idx) .* Hexcur(idx) *  exc2mmGain)) * amps(n);
        Vp = max(abs(H_xeqLoRms(idx) .* H_ieq(idx) * obj.VoltsPeakAmp)) * amps(n);
        
        % check mm and volt limits
        if (mm>MaxmmPk || Vp>MaxVPk)
            threshold = amps(n - 1);
            break;
        end
    end

    % determine gradient and y intercept
    m = (1 - 0)/(1 - threshold); % dy/dx
    b = 1 - m;
    
    % write output data to IO struct
    IO.lower = 0;
    IO.upper = 1;
    IO.gradient = m;
    IO.y_intercept = b;
    IO.threshold = threshold;
    
    % Figure out gain of an equivalent HPF with the same poles as poleXeq
    k_low = sqrt(real(poleXeq)^2 + imag(poleXeq)^2);                % TODO - need to look at in more detail
    k_low_inv = 1/k_low;

    % also in IO is this
    IO.k_low = k_low;
    IO.k_low_inv = k_low_inv;
    IO.k_alpha = 1.0;
    
    % plotting
    if obj.deqPlotFlgArr(3)
        y = m*amps + b;

        figure('Name', 'DEQ THRES: Input vs Output Levels', 'NumberTitle', 'off', 'windowstyle', 'docked')
        plot(db(amps),y)
        grid on
        xlabel('RMS Amplitude [dB FS]')
        ylabel('Low (0) -> High (1) RMS DEQ Filter')
        title('RMS Ampitude / Filter Type Determination')
        ylim([0 1])
    end     
    
def calcBassExtensionFlts(self,
                ftLow,
                Qt,
                maxMmPeak,
                maxVoltPeak,
                attackTime,
                releaseTime,
                rmsAttackTime,
                dropIeq,
                plot:bool=False):
    
    if plot:
        # create frequency vector
        # TODO - create log vector, and move into parent class
        fVec = np.linspace(10, fs/2, 10000)
    else:
        fVec = None
    
    # calculate alignment
    bAlignZ, aAlignZ = _calcAlignmentFlt(alignB, alignA, fs, fVec, plot)
    
    # calculate highpass shelf filter (used to create inductance filter)
    bShelf, aShelf, bHp, aHp = _calcHpShelfFlt(bAlignZ, aAlignZ, fs, fVec, plot)
    
    # create inducatance filter
    bIeq, aIeq = _createInductanceFlt(bShelf, aShelf, fs, fVec, plot)
    
    # create extension filter
    bXeq, aXeq = _createExtensionFlt(aHp, extFc, extQ, fs, fVec, plot)
    
    # TODO - impliment drop inductance filter
    
    # calculate displacement filter
    sos, gain, norm2mmGain = cef.calcExcursionFilter(self.driverParams['fVec'], self.driverParams['Hdisp'], self.driverParams['w0'],
                                                     self.driverParams['HdispGain'], self.driverParams['HdispMm'], filterType='lppeq',
                                                     enclosureType='sealed', fs=fs, plotData=False)
    
    # calculate excursion filter when for a maximum rms input level
    _calcMaxRmsXeqFilter(fVec, sos, gain, norm2mmGain, bIeq, aIeq, ftLow, Qt, self.driverParams['VoltsPeakAmp'],
                         maxMmPeak, maxVoltPeak, plotData=True)
    
    # TODO calculate attack release coefficients
    
    # calculate
    # _rms()
                
    if plot:
        plt.show()
    
if __name__ == "__main__":
    
    print('\nCalculating Bass Extension Parameters\n')
    
    # general parameters
    fs = 48000
    
    # measured params
    VoltsPeakAmp = 17.9 * np.sqrt(2)
    Bl = 5.184
    Mmc = 0.010
    Re = 4.7
    
    # tuning parameters
    ftLow = 40
    Qt = 0.65
    maxMmPeak = 1.4
    maxVoltPeak = 20 # TODO - needs implimenting
    attackTime = 0.001
    releaseTime = 0.100
    rmsAttackTime = 0.005
    dropIeq = False
    
    # initialise
    # TODO - what is volts peak, etc, doing here? not used?
    # lp = calcBassExtParams('codelibpython/dsp_bass/impedTestData/driverParams.npy', VoltsPeakAmp, Bl, Mmc, fs)
    lp = calcBassExtensionFlts('scratch/01_ALB_IMP_DEQ_reformatted_lp.mat', VoltsPeakAmp, Bl, Mmc, fs)
    
    # create parameter for a closed box
    params = lp.calcParams(ftLow, Qt, maxMmPeak, maxVoltPeak, attackTime, releaseTime, rmsAttackTime, dropIeq, plotData=True)
    
    print('\nFinished\n')
