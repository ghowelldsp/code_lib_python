#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

import numpy as np
import scipy.signal as sig
import scipy.optimize as opt
import matplotlib.pyplot as plt

class calcLumpedParams:
    
    def __init__(self, 
                 impedFileName:str,
                #  VoltsPeakAmp,
                #  Bl,
                #  Mmc,
                 Re:float=None):
        
        # import data
        self.impedFileName = impedFileName
        impedData = np.load(impedFileName, allow_pickle=True)
        
        # create empty data sets
        self.measuredData = {}
        self.estimatedParams = {}
        self.finalParams = {}
        self.driverParams = {}
        
        # assign local params
        self.measuredData['fVec'] = impedData['f']
        self.measuredData['Z'] = impedData['Z']
        # self.measuredData['VoltsPeakAmp'] = VoltsPeakAmp
        # self.measuredData['Bl'] = Bl
        # self.measuredData['Mmc'] = Mmc
        
        # if no Re value is input pick it from the lowest value of the measures impedance
        if (Re == None):
            freqMin = impedData['f'][0]
            ReMin = np.real(impedData['Z'][0])
            self.finalParams['Re'] = ReMin
            print(f'No Re value input, Re selected from lowest measured impedance')
            print(f'\t Re = {ReMin}')
            print(f'\t Freq = {freqMin}')
        else:
            self.finalParams['Re'] = Re
        
    def __findPeaks(self):
        
        # import local variables
        fVec = self.measuredData['fVec']
        ZReal = np.real(self.measuredData['Z'])
        
        # fit impedance to 3rd order polynominal
        polyCoeffs = np.polyfit(fVec, ZReal, deg=3)
        polyImp = np.polyval(polyCoeffs, fVec)
        
        # TODO - work out exactly what this is for
        if (polyImp[0] < np.abs(ZReal[0])):
            fittedImp = np.abs(ZReal - polyImp)
        else:
            fittedImp = np.abs(ZReal)
            
        # find the impedance peak
        peakLoc, _ = sig.find_peaks(fittedImp)
        self.peakLoc = peakLoc[0]
        
        # plot
        plt.figure()
        plt.plot(fVec, ZReal, label='Z real')
        plt.plot(fVec, polyImp, label='Z poly')
        plt.plot(fVec, fittedImp, '--', label='Z fitted')
        plt.plot(fVec[self.peakLoc], fittedImp[self.peakLoc], '*k')
        plt.legend()
        plt.grid()
        plt.title('Peak Finder')
        plt.ylabel('impedance [Ohms]')
        plt.xlabel('freq [Hz]')
        plt.xlim(fVec[0], fVec[-1])
        
    def __calcQmc(self):
        
        # import params
        fVec = self.measuredData['fVec']
        Z = self.measuredData['Z']
        f0 = self.estimatedParams['f0']
        
        # calculate the maximum impedance
        Zmax = np.abs(Z[self.peakLoc])
        
        # Q threshold
        QThres = 0.707 * Zmax
        
        # limit the impedance to frequencies that are less than twice the resonant frequecy
        ZLim = Z[fVec < (2 * f0)]
        
        # get the indexes of the impedance vector that are greater than the Q threshold level
        QIdx = np.where(ZLim >= QThres)[0]
        
        # calculate Qs
        # TODO - is this not Qms?
        if (len(QIdx) > 1):
            Qs = f0 / (fVec[QIdx[-1]] - fVec[QIdx[0]])
        else:
            Qs = 10
            
        self.estimatedParams['Qmc'] = Qs
        
    def __parameterEstimation(self):
        
        # import local variables
        fVec = self.measuredData['fVec']
        Z = self.measuredData['Z']
        
        # resonent frequency from the determined peak location
        f0 = fVec[self.peakLoc]
        w0 = 2 * np.pi * f0
        
        # impedance maximum
        Zmax = np.abs(Z[self.peakLoc])
        Res = Zmax - self.finalParams['Re']
        
        # limit freq vector to 500Hz if the resonent frequency is less than 400Hz, else limit to less than the double
        # the resonent frequency
        if f0 < 400:
            limVec = fVec < 500
        else:
            limVec = fVec < (2 * f0)
            
        # limit measurement data
        self.finalParams['fVec'] = fVec[limVec]                 # frequency vector 
        self.finalParams['wVec'] = 2 * np.pi * fVec[limVec]     # angular frequency vector
        self.finalParams['Z'] = Z[limVec]                       # complex impedance vector
        self.finalParams['S'] = 1j * self.finalParams['wVec']   # lapace vector
        
        # parameters estimated from the impedance vector
        self.estimatedParams['f0'] = f0
        self.estimatedParams['w0'] = w0
        self.estimatedParams['Res'] = Res
        
        # initial values for inductance (Le), and Eddy current resistance (R2) and inductance (L2)
        self.estimatedParams['Le'] = 0.01e-3
        self.estimatedParams['R2'] = 0.5
        self.estimatedParams['L2'] = 0.2e-3
        
        # calculte estimated Qmc
        self.__calcQmc()
        
    def __impedanceModel(self,
                         params,
                         flatDamping:bool=False):
        
        # extract params
        Res = params[0]
        w0 = params[1]
        Qmc = params[2]
        Le = params[3]
        R2 = params[4]
        L2 = params[5]
        
        # load variables
        S = self.finalParams['S']
        w = self.finalParams['wVec']
        
        # mechanical impedance
        if flatDamping:
            QmcTmp = Qmc * (w/w0)
            ResTmp = Res * (QmcTmp / Qmc)
            Zm = ResTmp / (1 + QmcTmp * (S/w0 + w0/S))
        else:
            Zm = Res / (1 + Qmc * (S/w0 + w0/S))
            
        # electrical impedance
        Ze = self.finalParams['Re'] + (S * Le) + S * L2 * R2 / (S * L2 + R2)
        
        # total impedance
        Z = Zm + Ze;
            
        return Z
        
    def __costFunction(self,
                       params):
        
        # model impedance
        Z = self.__impedanceModel(params)
        
        # square summed impedance error
        errVec = np.abs(Z - self.finalParams['Z']) / np.abs(self.finalParams['Z']); 
        sqSumErr = np.sum(errVec ** 2); 
        
        return sqSumErr
        
    def __parameterOptimisation(self):
        
        # create array of estimated parameters
        estParams = np.array([self.estimatedParams['Res'], 
                              self.estimatedParams['w0'], 
                              self.estimatedParams['Qmc'], 
                              self.estimatedParams['Le'],
                              self.estimatedParams['R2'],
                              self.estimatedParams['L2']])
        
        # optimise parameters according to mean-square error between the model and the measured impedance
        optData = opt.fmin(self.__costFunction, estParams, xtol=1e-10, ftol=1e-6, maxiter=10e3, maxfun=10e3, full_output=True)
        
        # add optimised parameters to class handle
        self.finalParams['Res'] = optData[0][0]
        self.finalParams['w0'] = optData[0][1]
        self.finalParams['Qmc'] = optData[0][2]
        self.finalParams['Le'] = optData[0][3]
        self.finalParams['R2'] = optData[0][4]
        self.finalParams['L2'] = optData[0][5]
        
        # calculate other parameters
        self.finalParams['Qec'] = self.finalParams['Re'] * self.finalParams['Qmc'] / self.finalParams['Res']
        self.finalParams['Qtc'] = 1 / (1/self.finalParams['Qmc'] + 1/self.finalParams['Qec'])
        
        # print estimated vs optimised parameters
        print('Estimated vs. Optimised Parameters')
        print(f'\t\t Estimated \n\t\t Optimised')
        print(f'\t Res \t {estParams[0]} \n\t\t {optData[0][0]}')
        print(f'\t w0 \t {estParams[1]} \n\t\t {optData[0][1]}')
        print(f'\t Qmc \t {estParams[2]} \n\t\t {optData[0][2]}')
        print(f'\t Le \t {estParams[3]} \n\t\t {optData[0][3]}')
        print(f'\t R2 \t {estParams[4]} \n\t\t {optData[0][4]}')
        print(f'\t L2 \t {estParams[5]} \n\t\t {optData[0][5]}')
        
        # calculate final impedance curve
        ZEst = self.__impedanceModel(estParams)
        ZFit = self.__impedanceModel(optData[0])
        
        # plot data
        fVec = self.finalParams['fVec']
        
        plt.figure()
        plt.plot(fVec, np.real(self.finalParams['Z']), label='measured')
        plt.plot(fVec, np.real(ZEst), '--', label='estimated')
        plt.plot(fVec, np.real(ZFit), '--', label='optimised')
        plt.legend()
        plt.grid()
        plt.title('Impedance Optimisation')
        plt.ylabel('impedance [Ohms]')
        plt.xlabel('freq [Hz]')
        plt.xlim(fVec[0], fVec[-1])
        
    def __alignmentExcursion(self):
        
        # load variables
        w0 = self.finalParams['w0']
        Qmc = self.finalParams['Qmc']
        S = self.finalParams['S']
        R2 = self.finalParams['R2']
        L2 = self.finalParams['L2']
        Le = self.finalParams['Le']
        Re = self.finalParams['Re']
        Qec = self.finalParams['Qec']
        
        # alignment params
        s2 = S ** 2
        w2 = R2 / L2
        we = Re / Le
        
        # determine alignment numerator and denominator coefficients
        bAlign = np.array([1/(w0**2 * w2), (1/w0)**2, 0, 0])
        aAlign = np.zeros([5])
        aAlign[0] = 1 / (w0**2 * w2 * we)
        aAlign[1] = (1/w0**2) * (1/we + L2/Re + 1/w2) + 1/(Qmc * w0 * we * w2)
        aAlign[2] = 1/(Qec*w0*w2) + 1/w0**2 + 1/(Qmc*w0*we) + 1/(Qmc*w0) * (1/w2 + L2/Re) + 1/(w2*we)
        aAlign[3] = 1/(Qec*w0) + 1/(Qmc*w0) + 1/we + 1/w2 + L2/Re
        aAlign[4] = 1
        
        # inductance transfer function
        Hind = (S/w2 + 1) / (s2/(w2*we) + S/we +  S*(L2/Re + 1/w2) + 1)

        # box transfer function
        Hbox = s2/w0**2 + S/(w0*Qmc) + 1

        # alignment                                                    
        fitAlign = (s2/w0**2) * Hind / (S/(w0*Qec) * Hind + Hbox)

        # displacement based on alignment only
        # TODO - should this not be /S
        fitDisplace = fitAlign / s2                                        

        # calculates excursion gain
        # TODO - get klippel parameters
        fitExcurGain = (self.measuredData['VoltsPeakAmp'] * self.measuredData['Bl']) / (Re * self.measuredData['Mmc'])

        # calculate excursion in mm
        fitExcurMm = fitDisplace * fitExcurGain * 1000
        
        # TODO - maybe create seperate alignment dictionary
        self.driverParams['bAlign'] = bAlign
        self.driverParams['aAlign'] = aAlign
        
    def __saveParams(self):
        
        # remove extension from input file name
        
        # TODO - update this
        np.save(f'impedTestData/driverParams', self.driverParams)
        
    def calcParams(self,
                   modelType:str):
        
        # TODO - add options for various model types
        
        self.__findPeaks()
        self.__parameterEstimation()
        self.__parameterOptimisation()
        self.__alignmentExcursion()
        self.__saveParams()
        
        plt.show()

if __name__ == "__main__":
    
    print('\nCalculating Lumped Parameters\n')
    
    # measured params
    VoltsPeakAmp = 15
    # Bl = 10
    # Mmc = 0.010
    # Re = 4.0
    
    # initialise
    lp = calcLumpedParams("impedTestData/testData.npz", Re)
    
    # create parameter for a closed box
    params = lp.calcParams("closed box")
    
    print('\nFinished\n')
    