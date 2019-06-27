#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Created on Thu Mar 21 00:01:22 2019

The relative power radiated into the nth harmonic of gravitational-wave

@author: Shucheng Yang
"""
#modules necessary
from scipy import special 
BesselJ = special.jn      #import Bessel function
def gne(n,e):
	'''
        The relative power radiated into the nth harmonic of gravitational-wave

        Shucheng Yang, Mar 2019

        Parameters
        ----------
        n : the nth harmonic
        e : eccentricity of orbit
 
        Returns
        -------
        g(n,e) : relative power of nth harmonic

        References
        ----------
		1.	Peters, P. C. & Mathews, J. Gravitational Radiation from Point Masses in a Keplerian Orbit. Physical Review 131, 435–440 (1963).

	'''
	g = (n**4*((4*BesselJ(n,e*n)**2)/(3.*n**2) + \
    (BesselJ(-2 + n,e*n) - 2*e*BesselJ(-1 + n,e*n) + (2*BesselJ(n,e*n))/n + \
    2*e*BesselJ(1 + n,e*n) - BesselJ(2 + n,e*n))**2 + \
    (1 - e**2)*(BesselJ(-2 + n,e*n) - 2*BesselJ(n,e*n) + BesselJ(2 + n,e*n))**2))/32.

	return g


if __name__ == '__main__':

#   Test
	n = 1            # nth harmonic
	e = 0.2          #eccentricity of orbit
	result = gne(n,e)
	print(result)
