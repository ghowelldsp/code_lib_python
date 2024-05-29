#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lumped Parameter Estimation

@author: G. Howell
"""

import numpy as np
import scipy.signal as sig
import scipy.optimize as opt
import matplotlib.pyplot as plt

import codelibpython.maths as math
        
def _findPeaks(fVec:np.array,
               Himp:np.array,
               plot:bool=False):
    """ Find Peaks
    
    Finds the peak of the measured impedance data.
    
    # TODO - understand what the polynominal fitting is for.

    Parameters
    ----------
    fVec : np.array [bins]
        Measured frequency vector [Hz]    
    Himp : np.array
        Measured impedance complex transfer function
    plot : bool, optional
        Plot data. Defaults to False.

    Returns
    -------
    impPeakIdx : int
        Index of the impedance peak.
    """
    
    # import local variables
    fVec = fVec
    HimpReal = np.real(Himp)
    
    # fit impedance to 3rd order polynominal
    polyCoeffs = np.polyfit(fVec, HimpReal, deg=3)
    HimpPoly = np.polyval(polyCoeffs, fVec)
    
    if (HimpPoly[0] < np.abs(HimpReal[0])):
        # TODO - work out exactly what this is for
        HimpFit = np.abs(HimpReal - HimpPoly)
    else:
        HimpFit = np.abs(HimpReal)
        
    # find the impedance peak
    impPeakIdx, _ = sig.find_peaks(HimpFit)
    impPeakIdx = impPeakIdx[0]
    
    if plot:
        plt.figure()
        plt.plot(fVec, HimpReal, label='Z real')
        plt.plot(fVec, HimpPoly, label='Z poly')
        plt.plot(fVec, HimpFit, '--', label='Z fitted')
        plt.plot(fVec[impPeakIdx], HimpFit[impPeakIdx], '*k')
        plt.legend()
        plt.grid()
        plt.title('Impedance Peak Finder')
        plt.ylabel('impedance [Ohms]')
        plt.xlabel('freq [Hz]')
        plt.xlim(fVec[0], fVec[-1])
    
    return impPeakIdx
        
def _calcQmc(fVec:np.array,
             Himp:np.array,
             HimpMax:float,
             f0:float,
             plot:bool=False):
    """ Calculate Qmc

    Calculates the value of Qmc which is seen as ratio of the frequency of the resonant impedance over the different 
    between the two frequency that lie -3 dB below the level of the resonant frequency. Seen as,
    
        Q = f0 / (f_-3dB[end] - f_-3dB[start])
        
    TODO - find out if the value is Qmc, Qs, etc?

    Parameters
    ----------
    fVec : np.array [bins]
        Measured frequency vector [Hz]    
    Himp : np.array
        Measured impedance complex transfer function
    HimpMax : float
        Value of the impedance of the resonant peak, the maximum impedance value.
    plot : bool, optional
        Plot data. Defaults to False.

    Returns:
    Qmc : float
        Value of Qmc.    
    """
    
    # limit the frequency vector
    fVecLimIdx = fVec < (2 * f0)
    
    # limit the impedance to frequencies that are less than twice the resonant frequecy
    Himp = Himp[fVecLimIdx]
    
    # q threshold
    qThres = 0.707 * HimpMax
    
    # get the indexes of the impedance vector that are greater than the Q threshold level
    # TODO - should it find the values that are above the abs(Himp)?
    qIdx = np.where(Himp >= qThres)[0]
    
    # calculate Q
    lowFreq = fVec[qIdx[0]]
    highFreq = fVec[qIdx[-1]]
    if (len(qIdx) > 1):
        Qmc = f0 / (highFreq - lowFreq)
    else:
        # TODO - seems like this could be a better index
        # TODO - raise a warning
        Qmc = 10
        
    if plot:
        plt.figure()
        plt.plot(fVec[fVecLimIdx], np.abs(Himp))
        plt.plot(np.array([lowFreq, highFreq]), np.array([qThres, qThres]), 'k--')
        plt.grid()
        plt.title('Qmc')
        plt.xlabel('freq [Hz]')
        plt.ylabel('magnitude [Ohms]')
    
    return Qmc
        
def _parameterEstimation(fVec:np.array,
                         Himp:np.array,
                         impPeakIdx:int,
                         Re:float,
                         plot:bool=False):
    """ Parameter Estimation

    Parameters
    ----------
    fVec : np.array [bins]
        Measured frequency vector [Hz]    
    Himp : np.array
        Measured impedance complex transfer function
    impPeakIdx : int
        Index of the impedance peak.
    Re : float
        Electrical DC resistance.
    plot : bool, optional
        Plot data. Defaults to False.

    Returns
    -------
    estimatedParams: dict
        Dictionary of all the parameters that are estimated before optimisation
    finalParams: dict
        Dictionary of all the final pararmeters that do not need optimising further.
    """
    
    # resonent frequency from the determined peak location
    f0 = fVec[impPeakIdx]
    w0 = 2 * np.pi * f0
    
    # impedance maximum
    HimpMax = np.abs(Himp[impPeakIdx])
    Res = HimpMax - Re
    
    # limit freq vector to 500Hz if the resonent frequency is less than 400Hz, else limit to less than the double
    # the resonent frequency
    if f0 < 400:
        fVecLimIdx = fVec < 500
    else:
        fVecLimIdx = fVec < (2 * f0)
        
    # limit measurement data
    finalParams = {}    
    finalParams['Re'] = Re
    finalParams['fVec'] = fVec[fVecLimIdx]              # frequency vector 
    finalParams['wVec'] = 2 * np.pi * fVec[fVecLimIdx]  # angular frequency vector
    finalParams['Himp'] = Himp[fVecLimIdx]              # complex impedance
    finalParams['sVec'] = 1j * finalParams['wVec']      # continuous domain vector
    
    # parameters estimated from the impedance vector
    estimatedParams = {}
    # estimatedParams['f0'] = f0 # TODO - don't think this is needed
    estimatedParams['w0'] = w0
    estimatedParams['Res'] = Res
    
    # initial values for inductance (Le), and Eddy current resistance (R2) and inductance (L2)
    estimatedParams['Le'] = 0.01e-3
    estimatedParams['R2'] = 0.5
    estimatedParams['L2'] = 0.2e-3
    
    # calculte estimated Qmc
    estimatedParams['Qmc'] = _calcQmc(fVec, Himp, HimpMax, f0, plot)
    
    return estimatedParams, finalParams
        
def _impedanceModel(params:np.array,
                    sVec:np.array,
                    wVec:np.array,
                    Re:float,
                    modelType:str='sealed',
                    flatDamping:bool=False):
    """ Impedance Model

    Parameters
    ----------
    params : np.array
        Parameters use for model prediction. Array of parameters in the order [Res, w0, Qmc, Le, R2, L2].
    sVec : np.array
        TODO
    wVec : np.array
        TODO
    Re : float
        Electrical resistance [Ohms].
    modelType : string
        Impedance model type. Defaults to 'sealed'.
    flatDamping : bool
        Enables / disables flat damping. Defaults to False.

    Returns
    -------
    Himp : np.array
        Calculated impedance data.
    """
    
    # extract params from array
    Res = params[0]
    w0 = params[1]
    Qmc = params[2]
    Le = params[3]
    R2 = params[4]
    L2 = params[5]
    
    # load variables
    S = sVec
    w = wVec
    
    if modelType == 'sealed':
        
        # mechanical impedance
        if flatDamping:
            QmcTmp = Qmc * (w/w0)
            ResTmp = Res * (QmcTmp / Qmc)
            HimpMech = ResTmp / (1 + QmcTmp * (S/w0 + w0/S))
        else:
            HimpMech = Res / (1 + Qmc * (S/w0 + w0/S))
            
        # electrical impedance
        HimpElec = Re + (S * Le) + S * L2 * R2 / (S * L2 + R2)
    
    elif modelType == 'ported':
        # TODO - implement
        pass
    
    # total impedance
    Himp = HimpMech + HimpElec;
            
    return Himp
        
def _costFunction(self,
                  params:np.array,
                  Himp:np.array,
                  sVec:np.array,
                  wVec:np.array,
                  Re:float,
                  modelType:str='sealed'):
    """ Cost Function
    
    The cost function used in the fitting of the impedance. Calculates the impedance based upon an electroacoustic
    model of the driver and enclosure, then determines the error between this fitted impedance and the measure 
    impedance.

    Parameters
    ----------
    params : np.array
        Parameters use for model prediction. Array of parameters in the order [Res, w0, Qmc, Le, R2, L2].
    Himp : np.array
        Measured impedance complex transfer function
    sVec : np.array
        TODO
    wVec : np.array
        TODO
    Re : float
        Electrical resistance [Ohms].
    modelType : string
        Impedance model type. Defaults to 'sealed'.

    Returns
    -------
    sqSumErr : float
        Square summed error value between the fitted and measured impedance data.
    """
    
    # model impedance
    HimpFit = self._impedanceModel(params, sVec, wVec, Re, modelType)
    
    # square summed impedance error
    errVec = np.abs(HimpFit - Himp) / np.abs(Himp); 
    sqSumErr = np.sum(errVec ** 2); 
    
    return sqSumErr
        
def _parameterOptimisation(estimatedParams:dict,
                           finalParams:dict,
                           modelType:str,
                           plot:bool=False):
    """ Parameter Optimisation
    
    Determines optimised lumped parameter values by minimising the error between the impedance determined from an 
    electroacoustic model of the driver and enclosure and the measured impedance.

    Parameters
    ----------
    estimatedParams: dict
        Dictionary of all the parameters that are estimated before optimisation
    finalParams: dict
        Dictionary of all the final pararmeters that do not need optimising further.
    modelType : string
        Impedance model type. Defaults to 'sealed'.
    plot : bool, optional
        Plot data. Defaults to False.

    Returns
    -------
    finalParams: dict
        Dictionary of all the final pararmeters that have been updated with the optimised values.
    """
    
    # create array of estimated parameters
    estParams = np.array([estimatedParams['Res'], 
                          estimatedParams['w0'], 
                          estimatedParams['Qmc'], 
                          estimatedParams['Le'],
                          estimatedParams['R2'],
                          estimatedParams['L2']])
    
    # optimise parameters according to mean-square error between the model and the measured impedance
    args  = (finalParams['Himp'], finalParams['sVec'], finalParams['wVec'], finalParams['Re'], modelType)
    optData = opt.fmin(_costFunction, estParams, args=args, xtol=1e-10, ftol=1e-6, maxiter=10e3, maxfun=10e3, full_output=True)
    
    # add optimised parameters to final params
    finalParams['Res'] = optData[0][0]
    finalParams['w0'] = optData[0][1]
    finalParams['Qmc'] = optData[0][2]
    finalParams['Le'] = optData[0][3]
    finalParams['R2'] = optData[0][4]
    finalParams['L2'] = optData[0][5]
    
    # calculate other parameters
    finalParams['Qec'] = finalParams['Re'] * finalParams['Qmc'] / finalParams['Res']
    finalParams['Qtc'] = 1 / (1/finalParams['Qmc'] + 1/finalParams['Qec'])
    
    if plot:
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
        fVec = finalParams['fVec']
        HimpEst = _impedanceModel(estParams)
        HimpFit = _impedanceModel(optData[0])
        
        # plot estimated impedance vs. fitted
        plt.figure()
        plt.semilogx(fVec, np.real(finalParams['Himp']), label='measured')
        plt.semilogx(fVec, np.real(HimpEst), '--', label='estimated')
        plt.semilogx(fVec, np.real(HimpFit), '--', label='optimised')
        plt.legend()
        plt.grid()
        plt.title('Impedance Optimisation')
        plt.ylabel('impedance [Ohms]')
        plt.xlabel('freq [Hz]')
        plt.xlim(fVec[0], fVec[-1])
        
    return finalParams
        
def _calcAlignment(finalParams:dict,
                   plot:bool=False):
    """ Calculate Alignment
    
    Calculate the alignment based upon the values found from the impedance.

    Parameters
    ----------
    finalParams: dict
        Dictionary of all the final pararmeters that do not need optimising further.
    plot : bool, optional
        Plot data. Defaults to False.

    Returns
    -------
    Halign : np.array
        Alignment transfer function.
    finalParams: dict
        Dictionary of all the final pararmeters that have been updated with the optimised values.
    """
    
    # load variables
    w0 = finalParams['w0']
    Qmc = finalParams['Qmc']
    S = finalParams['sVec']
    R2 = finalParams['R2']
    L2 = finalParams['L2']
    Le = finalParams['Le']
    Re = finalParams['Re']
    Qec = finalParams['Qec']
    
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
    Halign = (s2/w0**2) * Hind / (S/(w0*Qec) * Hind + Hbox)
    
    # update values in final params
    finalParams['bAlign'] = bAlign
    finalParams['aAlign'] = aAlign
    
    if plot:
        fVec = finalParams['fVec']
        plt.figure()
        plt.semilogx(fVec, 20*np.log10(Halign))
        plt.grid()
        plt.title('Alignment')
        plt.ylabel('magnitude [dB]')
        plt.xlabel('freq [Hz]')
        plt.xlim(fVec[0], fVec[-1])
    
    return Halign, finalParams
    
def _calcDisplacement(finalParams:dict,
                      Halign:np.array,
                      voltsPeakAmp:float,
                      Bl:float,
                      Mmc:float,
                      plot:bool=False):
    """ Calculate Displacement
    
    Calculates the complex displacement transfer function.

    Parameters
    ----------
    finalParams: dict
        Dictionary of all the final pararmeters that do not need optimising further.
    Halign : np.array
        Alignment transfer function.
    voltsPeakAmp : float
        Voltage at peak amplitude [V].
    Bl : float
        Force factor.
    Mmc : float
        Moving mass [g].
    plot : bool, optional
        Plot data. Defaults to False.

    Returns
    -------
    finalParams: dict
        Dictionary of all the final pararmeters that have been updated with displacement parameters.
    """

    # displacement based on alignment only
    # TODO - should this not be /S
    Hdisp = Halign / finalParams['sVec']**2

    # calculates displacement gain
    dispGain = (voltsPeakAmp * Bl) / (finalParams['Re'] * Mmc)

    # calculate displacement in mm
    HdispMm = Hdisp * dispGain * 1000
    
    # update values in final params
    finalParams['Hdisp'] = Hdisp
    finalParams['dispGain'] = dispGain
    finalParams['HdispMm'] = HdispMm
    
    if plot:
        fVec = finalParams['fVec']  
        plt.figure()
        plt.semilogx(fVec, np.abs(HdispMm)) # TODO - should this be real?
        plt.grid()
        plt.title('Displacement')
        plt.ylabel('displacement [mm]')
        plt.xlabel('freq [Hz]')
        plt.xlim(fVec[0], fVec[-1])
        
    return finalParams
        
def calcLumpedParams(fVec:np.array,
                     Himp:np.array,
                     voltsPeakAmp:float,
                     Bl:float,
                     Mmc:float,
                     Re:float=None,
                     modelType:str='sealed',
                     plot:bool=False):
    """ Calculate Lumped Parameters
    
    Calculates the lumped parameters.

    Parameters
    ----------
    fVec : np.array [bins]
        Measured frequency vector [Hz]    
    Himp : np.array
        Measured impedance complex transfer function
    voltsPeakAmp : float
        Voltage at peak amplitude [V].
    Bl : float
        Force factor.
    Mmc : float
        Moving mass [g].
    Re : float
        Electrical resistance [Ohms]. Note, if this is left as None then the resistance is calculated from the lowest
        frequency point from the measured impedance.
    plot : bool, optional
        Plot data. Defaults to False.

    Returns
    -------
    finalParams: dict
        Dictionary of all the final lumped parameter values.
    """
    
    # if no Re value is input pick it from the lowest value of the measures impedance
    if (Re == None):
        Re = np.real(Himp[0])
        print(f'No Re value input, Re selected from lowest measured impedance')
        print(f'\t Re = {Re}')
        print(f'\t Freq = {fVec[0]}')
    
    # process
    impPeakIdx = _findPeaks(fVec, Himp, plot)
    estimatedParams, finalParams = _parameterEstimation(fVec, Himp, impPeakIdx, Re, plot)
    finalParams =_parameterOptimisation(estimatedParams, finalParams, modelType, plot)
    Halign, finalParams =_calcAlignment(finalParams, plot)
    finalParams = _calcDisplacement(finalParams, Halign, voltsPeakAmp, Bl, Mmc, plot)
    
    return finalParams

if __name__ == "__main__":
    
    print('\nCalculating Lumped Parameters\n')
    
    # measured params
    voltsPeakAmp = 17.9 * np.sqrt(2)
    Bl = 5.184
    Mmc = 0.010
    Re = 4.7
    
    # load data impedance data from file
    impedData = np.load("impedTestData/01_ALB_IMP_DEQ_reformatted.npz", allow_pickle=True)
    
    # create parameter for a closed box
    driverParams = calcLumpedParams(impedData['f'], impedData['Z'], voltsPeakAmp, Bl, Mmc, Re=Re, plot=True)
    
    print('\nFinished\n')
    