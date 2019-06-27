#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 21:00:00 2019

The power spectual density(PSD) function of LISA

@author: Shucheng Yang
"""

#modules necessary
import numpy as np
# np.seterr(divide='ignore', invalid='ignore') #divide zero problem

def LISAcalPSD1(freqVec):
	'''
        The power spectual density(PSD) function2 of LISA 
        Shucheng Yang, Jun 2019
        Parameters
        ----------
        freqVec : frequency Vector, numpy array
 
        Returns
        -------
        LISAcalPSD1(freqVec) : PSD Vector, numpy array
        References
        ----------
		1. Babak, S., Fang, H., Gair, J. R., Glampedakis, K. & Hughes, S. A. Kludge’
		 gravitational waveforms for a test-body orbiting a Kerr black hole. Phys.
		 Rev. D 75, 024005 (2007).
	'''

	TAU = 50/3
	UTRANS = 0.25
	
	num = len(freqVec)

	u = 2 * np.pi * TAU * freqVec

	r = (1/ u**2) * ( (1 + np.cos(u)**2) * (1/3 - 2/u**2) + np.sin(u)**2 + 4*np.sin(u)*np.cos(u)/(u**3) )

	psd1 = (8.08e-48 / ((2*np.pi*freqVec)**4) + 5.52e-41)
	psd2 = (2.88e-48 / ((2*np.pi*freqVec)**4) + 5.52e-41) / r

	shadow1 =  np.zeros(num)
	shadow1[u < UTRANS] = 1

	shadow2 = np.zeros(num)
	shadow2[u >= UTRANS] = 1

	psd = psd1 * shadow1 + psd2 * shadow2

	return psd


def LISAcalPSD2(freqVec):
	'''
        The power spectual density(PSD) function2 of LISA 
        Shucheng Yang, Jun 2019
        Parameters
        ----------
        freqVec : frequency Vector, numpy array
 
        Returns
        -------
        LISAcalPSD2(freqVec) : PSD Vector, numpy array
        References
        ----------
		1. Sathyaprakash, B. S. & Schutz, B. F. Physics, Astrophysics and Cosmology with
		Gravitational Waves. Living Rev Relativ 12, 122004 (2009).
	'''

	S0 = 9.2e-44                      #unit psd(Hz^-1)	
	F0 = 1e-3                         #unit frequency(Hz)

	xVec = freqVec / F0
	psd = ((xVec/10)**(-4) + 173 + xVec**2) * S0 
	return psd


if __name__ == '__main__':
	F0 = 1e-3                         #unit frequency(Hz)
	num = 50
	xVec = 10**np.linspace(-2, 3, num)
	freqVec = xVec * F0
	psd1 = LISAcalPSD1(freqVec)
	psd2 = LISAcalPSD2(freqVec)
	print("psd1 = ",psd1)
	print("psd2 = ",psd2)

