#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multichannel AEC

Based on the paper "Multichannel Acoustic Echo Cancellation With Beamforming in Dynamic Environments" by Konforti, Cohen and Berdugo

@author: G. Howell
"""

import numpy as np
import matplotlib.pyplot as plt

def createArrayLocations(centerLocation:np.array,
                         nElements:int,
                         seperationDistance:float,
                         orientation:float):
    """ Create Array Locations
    
    Creates array locations in the cartisean plane, where the axes are in meters and zero degrees orientation is 
    'north' along the y axes.

                  y axes
                  0 deg
                    |
                    |
                    |
    -90 deg ----------------- +90 deg  x axes
                    |
                    |
                    |
                 +- 180 deg

    Parameters
    ----------
    centerLocation : np.array [axes]
        1D array of the center location in the cartesian plane. The first location is the x location and the second is 
        the y location.
    nElements : int
        The total number of elements in the array.
    seperationDistance : float
        Seperation distance in meters.
    orientation : float
        Orientation of the array in degrees. Zeros degrees in the instance is north, along the y vector.
        
    Return
    ------
    elementLocations : np.array
        Calculated element locations in the cartesian plane.
    """

    # convert the orientation to radians
    orientationRad = orientation / 180 * np.pi
    
    # find the location of the end element
    arrayLen = (nElements - 1) * seperationDistance
    elementLocations = np.zeros([nElements, 2])
    elementLocations[0,0] = centerLocation[0] - (arrayLen / 2) * np.cos(orientationRad) 
    elementLocations[0,1] = centerLocation[1] + (arrayLen / 2) * np.sin(orientationRad)
    
    # cartisian seperation distance between elements
    elementSeperationDistance = np.zeros([2])
    elementSeperationDistance[0] = seperationDistance * np.cos(orientationRad)
    elementSeperationDistance[1] = seperationDistance * np.sin(orientationRad)
    
    # calculate other element location
    for i in range(nElements-1):
        elementLocations[i+1,0] = elementLocations[i,0] + elementSeperationDistance[0]
        elementLocations[i+1,1] = elementLocations[i,1] - elementSeperationDistance[1]
        
    return elementLocations

def createLocations(plot:bool=True):
    
    # create driver locations
    driverCenterLocation = np.array([0.0,0.0])
    driverNElements = 5
    driverSeperationDistance = 0.1
    driverOrientation = 0
    driverLocations = createArrayLocations(driverCenterLocation, driverNElements, driverSeperationDistance, driverOrientation)
    
    # create mic locations
    micCenterLocation = np.array([0.0,0.0])
    micNElements = 2
    micSeperationDistance = 0.05
    micOrientation = 0
    micLocations = createArrayLocations(micCenterLocation, micNElements, micSeperationDistance, micOrientation)
    
    # create user locations
    userCenterLocation = np.array([0.0,1.0])
    userNElements = 1
    userSeperationDistance = 0.0
    userOrientation = 0
    userLocations = createArrayLocations(userCenterLocation, userNElements, userSeperationDistance, userOrientation)
    
    # plot locations
    if plot:
        plt.figure()
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.plot(driverLocations[:,0], driverLocations[:,1], '*', label='drivers', markersize=10)
        plt.plot(micLocations[:,0], micLocations[:,1], 'o', label='mic', markersize=10)
        plt.plot(userLocations[:,0], userLocations[:,1], 'D', label='user', markersize=10)
        plt.title('Driver, Mic and User Locations')
        plt.xlabel('X Location [m]')
        plt.ylabel('Y Location [m]')
        plt.legend()
        plt.grid()
        plt.show()
    
    return driverLocations, micLocations, userLocations
    
def calcGainsDelays(driverLocations:np.array,
                    userLocations:np.array,
                    micLocations:np.array,
                    speedOfSound:float=343):
    
    nMics = micLocations.shape[0]
    nUsers = userLocations.shape[0]
    nDrivers = driverLocations.shape[0]
    
    # calculate driver to mic distances - the form of the distances is [mics][speakers]
    driverMicDistances = np.zeros([nMics, nDrivers])
    userMicDistances = np.zeros([nMics, nUsers])
    for i in range(nMics):
        driverMicDistances[i,:] = np.abs(driverLocations - micLocations[i,:])
        userMicDistances[i,:] = np.abs(userLocations - micLocations[i,:])
        
    # calculate delays
    driverMicDelays = driverMicDistances / speedOfSound
    userMicDelays = userMicDistances / speedOfSound
    
    # calculate attenuation (gains)
    driverMicGains = 1 / driverMicDistances
    userMicGains = 1 / userMicDistances
    
    return driverMicDelays, driverMicGains, userMicDelays, userMicGains
    
def createImpulseResponse(delayTime:float,
                          gain:float):
    
    N = 100
    tVec = np.linspace(-10, 10, N)
    
    delayTime = 0.5
    sincResponse = np.sinc(tVec - delayTime)
    
    plt.figure()
    plt.plot(tVec, sincResponse)
    plt.grid()
    plt.show()
    
if __name__ == "__main__":

    # create driver, mic and user locations
    driverLocations, micLocations, userLocations = createLocations()
    
    # calculate delays and gains from the speaker and user locations to the mic locations
    calcGainsDelays(driverLocations, userLocations, micLocations)
    
    # create sinc functions
    # createSinc()
