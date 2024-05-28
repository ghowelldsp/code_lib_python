#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.signal._filter_design as fd

import designXeqIeqFilters as dxif
import designExcursionFilter as defi
import codelibpython.dsp_maths as dspm

class bassExtensionParams():
    
    def __init__(self,
                 driverParamsFileName,
                 VoltsPeakAmp,
                 Bl,
                 Mmc,
                 fs):
        
        # load driver params
        # driverParams = np.load(driverParamsFileName, allow_pickle=True)
        
        # # load variables
        # self.driverParams = {
        #     'fVec' : driverParams.item().get('fVec'),
        #     'w0' : driverParams.item().get('w0'),
        #     'bAlign' : driverParams.item().get('bAlign'),
        #     'aAlign' : driverParams.item().get('aAlign'),
        #     'Hdisp' : driverParams.item().get('Hdisp'),
        #     'HdispGain' : driverParams.item().get('HdispGain'),
        #     'HdispMm' : driverParams.item().get('HdispMm')
        # }
        
        # # set measured params
        # # TODO - think about moving these to the calculated lumped params script
        # self.driverParams['VoltsPeakAmp'] = VoltsPeakAmp
        # self.driverParams['Bl'] = Bl
        # self.driverParams['Mmc'] = Mmc
        
        from pymatreader import read_mat
        data = read_mat(driverParamsFileName)['impDataLumpParams']
        
        self.driverParams = {
            'fVec' : data['deqParams']['alignment']['freq'],
            'w0' : data['deqParams']['enclosure']['wc'],
            'bAlign' : data['deqParams']['alignment']['num2'],
            'aAlign' : data['deqParams']['alignment']['den2'],
            'Hdisp' : data['deqParams']['alignment']['excursion'],
            'HdispGain' : data['fitImpData']['excurGain'],
            'HdispMm' : data['fitImpData']['excurMm'],
        }
        
        self.driverParams['VoltsPeakAmp'] = VoltsPeakAmp
        
        # TODO - tidy up
        self.fs = fs
        
    def __writeParams(self):
        
        self.writeParams['extenFlt']['attackAlpha']
        self.writeParams['extenFlt']['attackOneMinusAlpha']
        
    def __calcMaxRmsXeqFilter(self,
                      fVec,
                      sos,
                      gain,
                      gainToMm,
                      bIeq,
                      aIeq,
                      fcLow,
                      Qt,
                      voltsPeakAmp,
                      maxDispLimit,
                      maxVoltLimit,
                      plotData:bool=True):
        """ Calculate Max RMS Extension Filter
        
        Typically, the desired cutoff frequency of the speaker response after the bass extention filter is applied 
        results in a displacement and voltage level that is over required limits when a maximum RMS signal level is
        input. In order to then determine the extension filter that should be applied to the signal cutoff frequency
        is iterativly increased untill both these limits are met. 

        Args:
            fVec (_type_): _description_
            sos (_type_): _description_
            gain (_type_): _description_
            gainToMm (_type_): _description_
            bIeq (_type_): _description_
            aIeq (_type_): _description_
            ftLow (_type_): _description_
            Qt (_type_): _description_
            voltsPeakAmp (_type_): _description_
            maxDispLimit (_type_): _description_
            maxVoltLimit (_type_): _description_
            plotData (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        
        # limit frequency array
        fVecLimIdx = (fVec > 10) & (fVec < 200)
        
        # displacement and ieq filters
        HdispMmTmp = gainToMm * gain * sig.sosfreqz(sos, fVec, fs=self.fs)[1]
        Hieq = sig.freqz(bIeq, aIeq, fVec, fs=self.fs)[1]
        
        # initialise variables
        maxDisp = np.inf
        maxVolts = np.inf
        fcHigh = fcLow
        
        # calculate displacement and voltage for increasing cutoff frequencies untill limits have been met
        while not((maxDisp <= maxDispLimit) and (maxVolts <= maxVoltLimit)):
            
            # iteratively increase cutoff of desired output repsonse
            fcHigh = fcHigh + 0.5;
            
            # calculate extension filter
            _, _, bXeqHigh, aXeqHigh = dxif.designXeqIeqFilters(self.driverParams['bAlign'], 
                                                                self.driverParams['aAlign'], 
                                                                fcHigh, Qt, self.fs)           
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
    
    # def __rmsThreshold(self):
        
        # % IEQ and XEQ low rms fitler tf's
        # [H_xeqLoRms] = freqz(bXeqLo, aXeqLo, freqArr, obj.fs);
        # [H_xeqHiRms] = freqz(bXeqHI, aXeqHI, freqArr, obj.fs);
        # [H_ieq] = freqz(bIeq, aIeq, freqArr, obj.fs);
        
        # % finds 2 index values of the frequency vector that are closest
        # % to 20Hz 
        # idx = find(freqArr >= 20, 2);
        
        # % array of amplitude values
        # amps = linspace(0, 1.2, 2000)';

        # % loop through amplitudes until A needs to start proctecting 
        # % the output (that will be the threshold)
        
        # % determines the max mm and max voltage at the various
        # % amplitude levels, stops once the max mm or volts level is 
        # % reached with the resulting amplitude being used as the 
        # % threshold
        # for n=2:length(amps)
            
        #     % calculate max mm and volts
        #     mm = max(abs(H_xeqLoRms(idx) .* H_ieq(idx) .* Hexcur(idx) *  exc2mmGain)) * amps(n);
        #     Vp = max(abs(H_xeqLoRms(idx) .* H_ieq(idx) * obj.VoltsPeakAmp)) * amps(n);
            
        #     % check mm and volt limits
        #     if (mm>MaxmmPk || Vp>MaxVPk)
        #         threshold = amps(n - 1);
        #         break;
        #     end
        # end

        # % determine gradient and y intercept
        # m = (1 - 0)/(1 - threshold); % dy/dx
        # b = 1 - m;
        
        # % write output data to IO struct
        # IO.lower = 0;
        # IO.upper = 1;
        # IO.gradient = m;
        # IO.y_intercept = b;
        # IO.threshold = threshold;
        
        # % Figure out gain of an equivalent HPF with the same poles as poleXeq
        # k_low = sqrt(real(poleXeq)^2 + imag(poleXeq)^2);                % TODO - need to look at in more detail
        # k_low_inv = 1/k_low;

        # % also in IO is this
        # IO.k_low = k_low;
        # IO.k_low_inv = k_low_inv;
        # IO.k_alpha = 1.0;
        
        # % plotting
        # if obj.deqPlotFlgArr(3)
        #     y = m*amps + b;

        #     figure('Name', 'DEQ THRES: Input vs Output Levels', 'NumberTitle', 'off', 'windowstyle', 'docked')
        #     plot(db(amps),y)
        #     grid on
        #     xlabel('RMS Amplitude [dB FS]')
        #     ylabel('Low (0) -> High (1) RMS DEQ Filter')
        #     title('RMS Ampitude / Filter Type Determination')
        #     ylim([0 1])
        # end
        
    
    def calcParams(self,
                   ftLow,
                   Qt,
                   maxMmPeak,
                   maxVoltPeak,
                   attackTime,
                   releaseTime,
                   rmsAttackTime,
                   dropIeq,
                   plotData:bool=False):
        
        # TODO - update plot
        bIeq, aIeq, bXeq, aXeq = dxif.designXeqIeqFilters(self.driverParams['bAlign'], 
                                 self.driverParams['aAlign'], 
                                 ftLow,
                                 Qt,
                                 fs,
                                 False)
        
        # TODO - impliment drop ieq
        
        sos, gain, norm2mmGain = defi.designExcursionFilter(self.driverParams['fVec'],
                                   self.driverParams['Hdisp'],
                                   self.driverParams['w0'],
                                   self.driverParams['HdispGain'],
                                   self.driverParams['HdispMm'],
                                   filterType='lppeq',
                                   enclosureType='sealed',
                                   fs=fs,
                                   plotData=False)
        
        
        # TODO - move to init
        fVec = np.linspace(10, fs/2, 10000)
        
        self.__calcMaxRmsXeqFilter(fVec,
                           sos,
                           gain,
                           norm2mmGain,
                           bIeq,
                           aIeq,
                           ftLow,
                           Qt,
                           self.driverParams['VoltsPeakAmp'],
                           maxMmPeak,
                           maxVoltPeak,
                           plotData=True)
        
        # TODO calculate ar coefficients
        
        # self.__rms()
            
        # self.__saveData()
                    
        if plotData:
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
    lp = calcBassExtParams('scratch/01_ALB_IMP_DEQ_reformatted_lp.mat', VoltsPeakAmp, Bl, Mmc, fs)
    
    # create parameter for a closed box
    params = lp.calcParams(ftLow, Qt, maxMmPeak, maxVoltPeak, attackTime, releaseTime, rmsAttackTime, dropIeq, plotData=True)
    
    print('\nFinished\n')
