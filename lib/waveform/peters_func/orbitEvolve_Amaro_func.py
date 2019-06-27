#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Created on Sun Mar 17 20:00:00 2019

Calculate the orbit coefficient e,a of EMRI or IMRI

@author: Shucheng Yang
"""

#modules necessary
import numpy as np  #import numpy, and name it as np
from astropy import constants as const #import constants 
import astropy.units as u              #import unit

from scipy.integrate import odeint     #import ordinary differential equation integrator

#modules used for test
import matplotlib.pyplot as plt        #import pyplot to plot


def deda_dt(M, m, e, a):
    """ 
        intermeidate function that return the da_dt, de_dt  

        Shucheng Yang, Mar 2019

        Parameters
        ----------
        M : the mass of central compact body
        m : the mass of small compact body
        e : eccentricity of orbit
        a : the semi-major axis of orbit

        Returns
        -------
        da_dt : the derivative of a(t)
        de_dt : the derivative of e(t)

        References
        ----------
        1.  Amaro-Seoane, P. Detecting Intermediate-Mass Ratio Inspirals From The Ground And Space. astro-ph.HE, 1–13 (2018).

    """
    G = (const.G).value
    C = (const.c).value
    da_dt = (-64/5) * (G**3 * M*m * (M+m)) * (1+ (73/24) * e**2 + (37/96) * e**4) / (C**5 * a**3 * (1-e**2)**(7/2)) 
    de_dt = (-304/15) * (G**3 * M * m * (M + m)) * (e * (1+ (121/304) * e**2)) / (C**5 * a**4 * (1-e**2)**(5/2))

    return de_dt, da_dt



#ordinary differential equation integrator
def evolve(M, m, e, a, timeVec):
    """ 
        ODE integrator that return the e Vector and a Vector

        Shucheng Yang, Mar 2019

        Parameters
        ----------
        M : the mass of central compact body
        m : the mass of small compact body
        e : initial eccentricity of orbit
        a : initial semi-major axis of orbit
        timeVec : array_like, the time sequence

        Returns
        -------
        e_Vec : array_like, the e sequence
        a_Vec : array_like, the a sequence

    """

    def odeg(vrb, t):#return de_dt, da_dt every step
        e, a = vrb.tolist()# a list that store the e_Vec and a Vec
        de_dt, da_dt = deda_dt(M, m, e, a)
        return de_dt,da_dt 

  
    init_status = e, a  #initial value
    args = () #Constants
    result = odeint(odeg, init_status, timeVec, args)# e_Vec and a Vec's list(n by 2 array)

    e_Vec = result[:,0] 
    a_Vec = result[:,1] 

    return e_Vec, a_Vec


if __name__ == '__main__':

#   Constants

    G = (const.G).value # gravitational constant
    C = (const.c).value # the speed of light

#   Parameters

    M = (1e6 * const.M_sun).to(u.kg).value  #the mass of central compact body, kilogram
    m = (1e1 * const.M_sun).to(u.kg).value  #the mass of small compact body, kilogram

    ########################
    R_unit = (G * M / C**2)#                #the unit length in G = C = 1 system, metre
    t_unit = R_unit / C    #                #the unit time in G = C = 1 system, second
    ########################

    e = 0.9                                 #eccentricity of orbit
    a = 30 * R_unit                         #the semi-major axis of orbit, metre

#   Time Vector
    ta = 0                                  #start time, G = C = 1 units
    tb = 17542 * 300 * t_unit               #end time
    step = t_unit                           #step

    timeVec = np.arange(ta, tb, step)       #time Vector(time sequence)


#   Results and show
    e_Vec, a_Vec = evolve(M, m, e, a, timeVec) #e Vector and a Vector
    a_Vec = a_Vec / R_unit                     #G = C = 1 units
    timeVec = timeVec / (1000 * t_unit)        #G = C = 1 units


    plt.figure(figsize=(10,8))
    #plot e and a
    plt.subplot(2,2,1)    
    plt.plot(e_Vec, a_Vec)
    plt.xlabel('e')
    plt.ylabel('a/M')

    #plot t and a
    plt.subplot(2,2,2)    
    plt.plot(timeVec, a_Vec)
    plt.xlabel('t/kM')
    plt.ylabel('a/M')

    #plot t and e
    plt.subplot(2,2,4)    
    plt.plot(timeVec, e_Vec)
    plt.xlabel('t/kM')
    plt.ylabel('e')
    plt.show()

