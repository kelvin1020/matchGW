#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Created on Thu Mar 21 10:25:00 2019
Generate evolved waveform of gravitational-wave(without dispersion)

@author: ysc
"""

##设置环境变量####################################################################################################################################################################################################################################################################################
#import sys
#import os 
#path_exe = ((sys.path[0]).split(os.sep))#导入当前目录
#path_exe.pop() 
#path_exe.pop()                   
#path_program = ((os.sep).join(path_exe)) 
#sys.path.append(path_program)           #将程序文件夹添加到环境变量
#path_now = sys.path[0]                  #当前目录
######################################################################################################################################################################################################################################################################################


#modules necessary
import numpy as np  #import numpy, and name it as np
from astropy import constants as const #import constants 
import astropy.units as u              #import unit


#power of harmonic
from lib.waveform.peters_func.powerNth_harmonic_Peters_func import gne
from lib.waveform.peters_func.orbitEvolve_Amaro_func import evolve  

#modules used for test
import matplotlib.pyplot as plt        #import pyplot to plot
#evolve

# from pycbc.types.timeseries import TimeSeries

from lib.waveform.cosmoModel import cosmoModel


# Constants
G = (const.G).value # gravitational constant
C = (const.c).value # the speed of light
 
class waveform_template(object):
    """docstring for waveform_template"""
    def __init__(self):
        self.name = None

class peters(waveform_template):
    """ 
        Generate evolved waveform of gravitational-wave(with dispersion)

        Shucheng Yang, Mar 2019

        Parameters
        ----------
        n : array_like, the nth harmonic
        nbase : the harmonic to be compared        
        M : the mass of central compact body
        m : the mass of small compact body
        e : eccentricity of orbit
        a : the semi-major axis of orbit
        Z : the redshit of gravitation-wave source
        D : the luminosity distance of gravitation-wave source
        D_ : not luminosity distance 
        lambda_g : the compton wavelength of graviton
        timedelay : a 2-D array  ( len(n) bytimeVec) )
   self.timeVec : array_like, a 1-D sequence of time 
        e_Vec : array_like, 1-D sequence of e 
        a_Vec : array_like, 1-D sequence of a

        Returns
        -------
        wave: array_like, the waveform of gravitational-wave(without dispersion), a complex sequence


        References
        ----------
        1.  Han, W.-B. Gravitational waves from extreme-mass-ratio inspirals in equatorially eccentric orbits. Int. J. Mod. Phys. D 23, 1450064–18 (2014).

        2.  Peters, P. C. & Mathews, J. Gravitational Radiation from Point Masses in a Keplerian Orbit. Physical Review 131, 435–440 (1963).

        3.  Mirshekari, S., Yunes, N. & Will, C. M. Constraining Lorentz-violating, modified dispersion relations with gravitational waves. Phys. Rev. D 85, 024041 (2012).       
    """
    def __init__(self, **kwargs):
        super(waveform_template, self).__init__()
        self.name = "peters"
        
        self.n = kwargs["n"]
        self.M = kwargs["M"]
        self.m = kwargs["m"]

        self.e = kwargs["e"]
        self.a = kwargs["a"]
        self.D = kwargs["D"]

        self.cosmo= cosmoModel()
        self.Z = self.cosmo.z_accu(self.D)    
        self.D_a = self.cosmo.D_a(self.Z)

    def calculate(self, duration =  65536, delta_t = 0.25):
        self.duration = duration
        self.delta_t  = delta_t
        self.timeVec  = np.arange(0, self.duration, self.delta_t)

        #   E and a Vector
        self.e_Vec, self.a_Vec = evolve(self.M, self.m, self.e, self.a, self.timeVec)
     
            
        self.h = np.array([gne(ni, self.e_Vec) * G**2 * self.M * self.m / (self.D * self.a_Vec * C**4) for ni in self.n])    
        self.omega_n = np.array([ni * np.sqrt(G * self.M/ self.a_Vec**3) for ni in self.n])   
        # See Ref.2 III (1)

        wave = np.array([self.h[i] *  np.exp((self.omega_n[i] * self.timeVec) * -1j) for i in range(0,len(self.h))]) 
        wave = np.sum(wave, axis=0) 
        # See Ref.1 (48)
      
        self.hplus = wave.real
        self.hcross  = - wave.imag

    def add_dispersion(self, lambda_g, nbase=6): 
        self.lambda_g = lambda_g   
        self.nbase = nbase

        f_n = self.omega_n / (2 * np.pi)

        omega_base = self.nbase * np.sqrt(G * self.M/ self.a_Vec**3) 
        f_base = omega_base / (2 * np.pi)

        timedelay = np.array([(1 + self.Z) * C * self.D_a /(2 * self.lambda_g**2) * (1 / f_ni**2 - 1 / f_base**2 ) for f_ni in f_n])
        # See Ref.3 III (14)


        wave = np.array([self.h[i] *  np.exp((self.omega_n[i] * (self.timeVec + timedelay[i])) * -1j) for i in range(0,len(self.h))]) 
        wave = np.sum(wave, axis=0) 
        # See Ref.1 (48)

        self.hplus = wave.real# wave.real
        self.hcross  = - wave.imag


if __name__ == '__main__':

#   Parameters
    n = np.arange(1,11,1)                   #the nth harmonic
    print(n)


    M = (1e6 * const.M_sun).to(u.kg).value  #the mass of system, kilogram
    m = (1e1 * const.M_sun).to(u.kg).value  #the mass of small compact body, kilogram

    ########################
    R_unit = (G * M / C**2)#                #the unit length in G = C = 1 system, metre
    t_unit = R_unit / C    #                #the unit time in G = C = 1 system, second
    ########################

    e = 0.5                                 #eccentricity of orbit
    p = 12 * R_unit                         #semi-latus rectum

    a = p / (1 - e**2)                      #the semi-major axis of orbit, metre

    D = (1.00 * u.Gpc).to('m').value

#   Time Vector                  
    tb = 17542  * t_unit                 #end time, second
    step = t_unit                           #step

    wave = peters(n = n, M = M, m = m, e = e, a = a, D = D)
    wave.calculate(duration = tb, delta_t = step)

    h_plus_Vec = wave.hplus                  # h_plus of gravitational-wave
    h_cross_Vec = wave.hcross                # h_cross of gravitational-wave

#   dispersion
    lambda_g = (1.6e13 * u.km).to('m').value#the compton wavelength of graviton
    wave.add_dispersion(lambda_g)

    h_plus_d_Vec = wave.hplus                  # h_plus of gravitational-wave (dispersion)
    h_cross_d_Vec = wave.hcross                # h_cross of gravitational-wave (dispersion)


    a_Vec = wave.a_Vec           
    a_Vec = a_Vec / R_unit                     #G = C = 1 units

    e_Vec = wave.e_Vec
    p_Vec = a_Vec * (1 - wave.e_Vec**2)          #G = C = 1 units

    timeVec = wave.timeVec  
    vecLength = len(timeVec)

    plt.figure(figsize=(10,8))
    #plot t and e
    plt.subplot(2,2,1)    
    plt.plot(timeVec, e_Vec)
    # plt.xlabel('t/kM')
    plt.xlabel('t/s')    
    plt.ylabel('e')

    #plot t and p
    plt.subplot(2,2,2)    
    plt.plot(timeVec, p_Vec)
    # plt.xlabel('t/kM')
    plt.xlabel('t/s')    
    plt.ylabel('p/M')

    #plot t and h+,hx (only part of waveform)
    plt.subplot(2,2,(3,4))    
    plt.plot(timeVec[0:int(vecLength/8)], h_plus_Vec[0:int(vecLength/8)],label = '$h_+$') 
    plt.plot(timeVec[0:int(vecLength/8)], h_plus_d_Vec[0:int(vecLength/8)],label = '$hd_{+}$') 

    # plt.plot(timeVec[0:int(vecLength/8)], h_cross_Vec[0:int(vecLength/8)],label = '$h_x$')
    # plt.plot(timeVec[0:int(vecLength/8)], h_cross_d_Vec[0:int(vecLength/8)],label = '$hd_{x}$')
    # plt.xlabel('t/kM')
    plt.xlabel('t/s')  
    plt.ylabel('h')
    plt.legend(loc ='best')
    plt.show()





