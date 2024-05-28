#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bass Extension

@author: G. Howell
"""

def bassExtension():
    
    def __init__(self):
        
        pass

    # def checkImpedance(self,
    #                    filename:str):
        
    def calcLumpedParams(self,
                         filename:str,
                         plot:bool=False,
                         writeData:bool=False):
        
        # import data
        impedData = np.load(filename, allow_pickle=True)
        
        # calculate parameters
        calcLumpedParams(impedData['f'], impedData['Z'], plot)
        
    # def calcTuningParams():
    
if __name__ == "__main__":
    
    print('\nCalculating Bass Extension Parameters\n')

    print('\nFinished\n')
