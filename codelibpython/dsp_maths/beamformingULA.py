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
    
    def __init__(self, d, nElements, c=343, fs=48000):
        """ Calculate Delays
        
        Calculates the delays values for the relative distances.

        Args:
            d (number):             Seperation distance between adjacent array elements [m].
            nElements (string):     Number of elements in array.
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
        self.modelType = 'planeWave'
        
        # calculate element locations referenced to the center of the array
        firstLoc = -(nElements - 1) * d/2
        elementLocs = np.linspace(firstLoc, -firstLoc, nElements)
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
        
        # calculate time delays (note: negate the distance to return correction delays)
        delays = distances / self.c
        self.timeDelays = delays
        
        if delayType == 'samples':
            # calculate samples delays
            delays = delays * self.fs
    
        return delays
    
    def __sphericalWave(self, steeringAngle, delayType='time'):
        """ Spherical Wave Delay Models
        
        Calculates the phase delay relative to the center of the array using a plane wave model

        Args:
            steeringAngle (float):  Steering angle [deg]
            delayType (string):     Type of delays to return. Options are 'time' and 'samples'.

        Returns:
            delays (np.array):      2D matrix of delays in terms of angles and elements, taking the form of 
                                    [angles, elements].
        """
        
        # calculate driver x-y coordinates
        elementXYCoords = np.zeros((self.nElements, 2))
        elementXYCoords[:,0] = self.elementLocs
        
        # listener location
        steeringAngle = np.reshape(steeringAngle / 180 * np.pi, [-1])
        listenerXYCoords = np.array((np.sin(steeringAngle), np.cos(steeringAngle)))
        listenerXYCoords = listenerXYCoords.T
        
        # calculate the Euclidean distance between each driver and the listener location
        nAngles = len(steeringAngle)
        distances = np.zeros([nAngles, self.nElements])
        for i in range(nAngles):
            distancesTmp = listenerXYCoords[i,:] - elementXYCoords
            distances[i,:] = np.sqrt(np.sum(distancesTmp**2, axis=1)) - 1
        
        # calculate delays
        delays = self.__calcDelays(distances, delayType)
        
        return delays
    
    def __planeWave(self, steeringAngle, delayType='time'):
        """ Plane Wave Delay Models
        
        Calculates the phase delay relative to the center of the array using a plane wave model

        Args:
            steeringAngle (float):  Steering angle [deg]
            delayType (string):     Type of delays to return. Options are 'time' and 'samples'.

        Returns:
            delays (np.array):      2D matrix of delays in terms of angles and elements, taking the form of 
                                    [angles, elements].
        """
        
        # relative distances
        distances = self.elementLocs * np.sin(np.reshape(steeringAngle / 180 * np.pi, [-1,1]))
        
        # calculate distance from relative to the furthest element
        distances += np.abs(distances[0,0])
        
        # calculate delays
        delays = self.__calcDelays(distances, delayType)
        
        return delays
    
    def process(self, steeringAngle, delayType='time', modelType='planeWave'):
        """ Process
        
        Calculates the delays using a given model.

        Args:
            steeringAngle (float):  Steering angle [deg]
            delayType (string):     Type of delays to return. Options are 'time' and 'samples'.
            modelType (string):     Type of model to calculate delays for. Options are 'planeWave' and 
                                    'sphericalWave'.

        Returns:
            delays (np.array):      2D matrix of delays in terms of angles and elements, taking the form of 
                                    [angles, elements].
        """
        
        self.modelType = modelType
        
        if (modelType == 'planeWave'):
            delays = self.__planeWave(steeringAngle, delayType)
        elif (modelType == 'sphericalWave'):
            delays = self.__sphericalWave(steeringAngle, delayType)
            
        return delays
    
    def plot(self, f):
        """ Plotting
        
        Plot polar response for specified frequencies
        
        Args:
            f (float):     Frequency [Hz]
        """
        
        # calculate delays between -90 and 90 degrees and offset against steering angle (note: the return delays are
        # negated to get actual delays and not correction delays values)
        plotAngles = np.arange(-90,90,1)
        timeOffset = self.timeDelays
        timeDelays = -self.process(plotAngles, 'time', self.modelType) + timeOffset

        # calculate summed magnitude values
        mag = np.absolute(np.sum(np.exp(1j*2*np.pi*f*timeDelays), axis=1)) / self.nElements
        
        # plot
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(plotAngles/180*np.pi, 20*np.log10(mag))
        ax.grid(True)
        ax.set_title(f'Polar Response - Freq = {f} Hz')
        ax.set_xlabel('Magnitude [dB]')
        ax.set_thetamin(-90)
        ax.set_thetamax(90)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        plt.show()

if __name__ == "__main__":
    
    # model parameters
    seperationDistance = 0.2            # seperation distance between elements [m]
    nElements = 5                       # number of elements
    steeringAngle = 45                  # steering angle [deg]
    
    # initialise the beamformer
    bfH = beamformingULA(seperationDistance, nElements)
    
    # calculate plane wave model
    pwTimeDelays = bfH.process(steeringAngle, 'time', 'planeWave')
    pwSampleDelays = bfH.process(steeringAngle, 'samples', 'planeWave')
    
    # plotting
    frequency = 200
    bfH.plot(frequency)
    
    # calculate spherical wave model
    swTimeDelays = bfH.process(steeringAngle, 'time', 'sphericalWave')
    swSampleDelays = bfH.process(steeringAngle, 'samples', 'sphericalWave')
    
    # plotting
    frequency = 1000
    bfH.plot(frequency)
    