#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 21:00:00 2019

The power spectual density(PSD) function of LISA

@author: Shucheng Yang
"""

#modules necessary
import numpy as np
# np.seterr(divide='ignore', invalid='ignore') #divide zero problem

def B_DECIGOcalPSD(freqVec):
    '''
        The power spectual density(PSD) function2 of LISA 
        Shucheng Yang, July 2020
        Parameters
        ----------
        freqVec : frequency Vector, numpy array
 
        Returns
        -------
        LISAcalPSD1(freqVec) : PSD Vector, numpy array
        References
        ----------
        1. Multiband Gravitational-Wave Astronomy:Observing binary inspirals with a decihertz detector, B-DECIGO (20)
        https://arxiv.org/pdf/1802.06977.pdf
    '''

    #remove 0
    if (freqVec[0] == 0):
        freqVec[0] = 1e-30
    #
    
    S0 = 4.040e-46                      #unit psd(Hz^-1)  
    F0 = 1e0                           #unit frequency(Hz)

    xVec = freqVec / F0
    
    return  (1 + 1.584e-2 * xVec**(-4.0) + 1.584e-3 * xVec**2.0) * S0 

def DECIGOcalPSD(freqVec):
    '''
        The power spectual density(PSD) function2 of LISA 
        Shucheng Yang, July 2020
        Parameters
        ----------
        freqVec : frequency Vector, numpy array
 
        Returns
        -------
        LISAcalPSD1(freqVec) : PSD Vector, numpy array
        References
        ----------
        1. Detector configuration of DECIGO/BBO and identification of cosmological neutron-star binaries (5)
        https://arxiv.org/pdf/1101.3940.pdf
    '''

    #remove 0
    if (freqVec[0] == 0):
        freqVec[0] = 1e-30
 
    F0 = 1e0                           #unit frequency(Hz)
    Fp = 7.36                        
   
    return 7.05e-48 * ( 1 + (freqVec / Fp)**2 )  + 4.8e-51 * (freqVec / F0)**(-4) * (1 / ( 1 + (freqVec / Fp)**2 )) + \
    5.33e-52 * (freqVec / F0)**(-4)
    



if __name__ == '__main__':
    F0 = 1e0                         #unit frequency(Hz)
    num = 50
    xVec = 10.0**np.linspace(-2, 3, num)
    freqVec = xVec * F0
    psd1 = B_DECIGOcalPSD(freqVec)
    print("psd1 = ",psd1)
    
    psd2 = DECIGOcalPSD(freqVec)
    print("psd2 = ",psd2)

