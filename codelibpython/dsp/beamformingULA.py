#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

import numpy as np
import matplotlib.pyplot as plt

class beamformingULA:
    """
    Beamforming Uniform Linear Array
    
    Calculate the phase delay associated with a uniform linear arrays. Delays can be calculated by using either a place
    wave model, or a spherical wave model and can return the delay values in either time or samples.
    
    """
    
    def __init__(self, d, nElements, angles, c=343, fs=48000):
        """ Calculate Delays
        
        Calculates the delays values for the relative distances.

        Args:
            d (number):             Seperation distance between adjacent array elements [m].
            nElements (string):     Number of elements in array.
            angles (np.array):      1D array of angles to calculate delays over [degrees].
            c (number):             Speed of sound [m/s]. Defaults to the speed of sound at room temperature if not
                                    input.
            fs (number):            Sample rate [Hz]. Only required in the return delays values are calcualted in 
                                    samples. If not input it defaults to 48kHz.

        Returns:
            delays (np.array):      2D matrix of delays in terms of angles and elements, taking the form of 
                                    [angles, elements].
        """
        self.d = d
        self.nElements = nElements
        self.c = c
        self.fs = fs
        
        # convert angles list to numpy array in radians
        self.angles = np.array(angles) / 180 * np.pi
        
        # calculate element locations referenced to the center of the array
        elementLocs = d * (nElements - np.linspace(1, nElements*2-1, nElements)) / 2
        self.elementLocs = np.array(elementLocs, ndmin=2)
        
    def __calcDelays(self, distances, delayType):
        """ Calculate Delays
        
        Calculates the delays values for the relative distances.

        Args:
            distances (np.array):   2D matrix of distances in terms of angles and elements, taking the form of 
                                    [angles, elements].
            delayType (string):     Type of delays to return. Options are 'time' and 'samples'.

        Returns:
            delays (np.array):      2D matrix of delays in terms of angles and elements, taking the form of 
                                    [angles, elements].
        """
        
        # calculate time delays
        delays = distances / self.c
        self.timeDelays = delays
        
        if delayType == 'samples':
            # calculate samples delays
            delays = delays * self.fs
    
        return delays
    
    def sphericalWave(self, delayType='time'):
        """ Spherical Wave Delay Models
        
        Calculates the phase delay relative to the center of the array using a plane wave model

        Args:
            delayType (string):     Type of delays to return. Options are 'time' and 'samples'.

        Returns:
            delays (np.array):      2D matrix of delays in terms of angles and elements, taking the form of 
                                    [angles, elements].
        """
        
        # calculate driver x-y coordinates
        elementXYCoords = np.zeros((self.nElements, 2))
        elementXYCoords[:,0] = self.elementLocs
        
        # listener location
        listenerXYCoords = np.array((np.sin(self.angles), np.cos(self.angles)))
        listenerXYCoords = listenerXYCoords.T
        
        # calculate the eucliean distance between each driver and the listener location
        distances = np.zeros([len(self.angles), self.nElements])
        for i in range(len(self.angles)):
            distancesTmp = listenerXYCoords[i,:] - elementXYCoords
            distances[i,:] = np.sqrt(np.sum(distancesTmp**2, axis=1)) - 1
        
        # calculate delays
        delays = self.__calcDelays(distances, delayType)
        
        return delays
    
    def planeWave(self, delayType='time'):
        """ Plane Wave Delay Models
        
        Calculates the phase delay relative to the center of the array using a plane wave model

        Args:
            delayType (string):     Type of delays to return. Options are 'time' and 'samples'.

        Returns:
            delays (np.array):      2D matrix of delays in terms of angles and elements, taking the form of 
                                    [angles, elements].
        """
        
        # relative distances
        distances = self.elementLocs * np.sin(np.reshape(self.angles, [-1,1]))
        
        # calculate delays
        delays = self.__calcDelays(distances, delayType)
        
        return delays
    
    def plot(self, f):
        """ Plotting
        
        Plot polar response for specified frequecies
        
        Args:
            f (number):     Frequency [Hz]
        """

        # calculate summed magnitude values
        mag = np.absolute(np.sum(np.exp(1j*2*np.pi*f*self.timeDelays), axis=1)) / self.nElements
        
        # plot
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(self.angles, 20*np.log10(mag))
        ax.grid(True)
        ax.set_title(f'Polar Response - Freq = {f} Hz')
        ax.set_xlabel('Magnitude [dB]')
        ax.set_thetamin(-90)
        ax.set_thetamax(90)
        ax.set_theta_zero_location('N')
        
        plt.show()

if __name__ == "__main__":
    
    # model parameters
    seperationDistance = 0.2            # seperation distance between elements [m]
    nElements = 5                       # number of elements
    angles = np.arange(-90,90,1)        # angles to calculate delays over [deg]
    
    # initialise the beamformer
    bfH = beamformingULA(seperationDistance, nElements, angles)
    
    # calculate plane wave model
    pwTimeDelays = bfH.planeWave('time')
    pwSampleDelays = bfH.planeWave('samples')
    
    # plotting
    frequency = 1000
    bfH.plot(frequency)
    
    # calculate spherical wave model
    swTimeDelays = bfH.sphericalWave('time')
    swSampleDelays = bfH.sphericalWave('samples')
    
    # plotting
    frequency = 1000
    bfH.plot(frequency)
    