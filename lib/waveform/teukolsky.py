#!/usr/bin/env python
# -*- coding: utf-8 -*- 
"""
Created on Thu Mar 21 10:25:00 2019
Generate evolved waveform of gravitational-wave(without dispersion)

@author: ysc
"""
##设置环境变量####################################################################################################################################################################################################################################################################################
import sys
import os 
path_exe = ((sys.path[0]).split(os.sep))#导入当前目录
path_exe.pop() 
path_exe.pop()                   
path_program = ((os.sep).join(path_exe)) 
sys.path.append(path_program)           #将程序文件夹添加到环境变量
path_now = sys.path[0]                  #当前目录
######################################################################################################################################################################################################################################################################################



#modules necessary
import numpy as np  #import numpy, and name it as np
from astropy import constants as const #import constants 
import astropy.units as u              #import unit


#power of harmonic
from multiprocessing import Pool


#from lib.waveform.peters_func.orbitEvolve_hacc8 import evolve  
from lib.waveform.teukolsky_func.orbitEvolve_hacc8_func import evolve
from lib.waveform.teukolsky_func.harmonic import harmonic as harmonic
from lib.waveform.teukolsky_func.omega import wmk, wmk_uniarg


#modules used for test
import matplotlib.pyplot as plt        #import pyplot to plot
#evolve

# from pycbc.types.timeseries import TimeSeries

from lib.waveform.cosmoModel import cosmoModel

# uharmonic = np.frompyfunc(harmonic, 7, 2)
import time

# Constants
G = (const.G).value # gravitational constant
C = (const.c).value # the speed of light


wmk_vc = np.vectorize(wmk)



class waveform_template(object):
    """docstring for waveform_template"""
    def __init__(self):
        self.name = None

class teukolsky(waveform_template):
    """ 
        Generate evolved waveform of gravitational-wave(with dispersion)

        Shucheng Yang, Mar 2019

        Parameters
        ----------
        atilde: spiral a
        l : array_like, the nth harmonic
        k : array_like, the nth harmonic

        kbase : the harmonic to be compared        
        M : total Mass
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
        self.name = "teukolsky"
        self.atilde = kwargs["atilde"]
        self.nu = kwargs["nu"]
        
        self.M = kwargs["M"]
        self.m = 1.0/2.0 * self.M * (1 - np.sqrt(1 - 4 * self.nu))



        self.l = kwargs["l"]
        self.k = kwargs["k"]

        self.e = kwargs["e"]
        self.a = kwargs["a"]
        self.D = kwargs["D"]

        self.p = self.a * (1-self.e**2)

        self.cosmo= cosmoModel()
        self.Z = self.cosmo.z_accu(self.D)    
        self.D_a = self.cosmo.D_a(self.Z)

        self.R_unit = (G * self.M / C**2)#                #the unit length in G = C = 1 system, metre
        self.t_unit = self.R_unit / C    #                #the unit time in G = C = 1 system, second

    def calculate(self, duration, delta_t):
        self.duration = duration
        self.delta_t  = delta_t
        self.timeVec  = np.arange(0, self.duration, self.delta_t)
        timeVec  = np.arange(0, self.duration/self.t_unit, self.delta_t/self.t_unit)

        ans = np.array([harmonic(self.atilde, self.nu, self.p/ self.R_unit, self.e, l, l, self.k) for l in self.l])
        ans = np.vstack([item for item in ans])
        
        hlmk = ans[:, 0:1]  
        self.omega_mk = ans[:, 1:2] 
        # See Ref.2 III (1)

        self.n = len(timeVec)   #row n
        self.cn = len(self.omega_mk)   #column m
        self.caltimeVec = (timeVec).reshape(1, self.n)

        self.hlmk = np.tile(hlmk, self.n)
        self.factor = 1#(self.D / self.R_unit) 恢复公有制下大小
        wave = (2.0 / self.factor ) *  self.hlmk * np.e**(-1j * self.omega_mk@self.caltimeVec)

        wave = np.sum(wave, axis = 0)

        # See Ref.1 (48)
      
        self.hplus = wave.real
        self.hcross  = - wave.imag

    def add_dispersion(self, lambda_g, kbase=4): 
        self.lambda_g = lambda_g   
        self.kbase = kbase

        f_mk = self.omega_mk / (2 * np.pi)

        f_base = f_mk[int(self.kbase - min(self.k) )]

        timedelay = np.array([(1 + self.Z) * (self.D_a / self.R_unit) / (2 * (self.lambda_g / self.R_unit)**2) * (1 / f_mki**2 - 1 / f_base**2 ) for f_mki in f_mk]) #G = c = 1
        timedelay  = np.tile(timedelay, self.n)

        caltimeVec = np.tile(self.caltimeVec, (self.cn, 1))
        omega_mk = np.tile(self.omega_mk, (1, self.n)) 

        # # See Ref.3 III (14)

        wave = (2.0 / self.factor ) *  self.hlmk * np.e**(-1j * omega_mk * (caltimeVec + timedelay))
        wave = np.sum(wave, axis = 0)
        # # # See Ref.1 (48)

        self.hplus = wave.real# wave.real
        self.hcross  = - wave.imag


    def ecalculate(self, duration, delta_t):
        self.duration = duration
        self.delta_t  = delta_t
        self.timeVec  = np.arange(0, self.duration, self.delta_t)
        timeVec  = np.arange(0, self.duration/self.t_unit, self.delta_t/self.t_unit)
        self.n = len(timeVec)   #row n

        # evolution
        vm =  np.sqrt(1/(self.p/ self.R_unit))
        timea = time.time()
        self.e_Vec, vm_Vec =  evolve(self.atilde, self.nu, self.e, vm, timeVec) #e Vector and a Vector
        timeb = time.time()
        print("evolution finished, cost%6.3fs"%(timeb - timea))

        pm_Vec =  1 / vm_Vec**2                    #G = C = 1 units
        self.p_Vec = pm_Vec * self.R_unit
        self.a_Vec = self.p_Vec / (1 - self.e_Vec**2)

        # omega_mk 
        timea = time.time()
        #evolution
        omega_mk_Vec = np.array( [(np.array([wmk(self.atilde, self.nu, self.p_Vec[T]/ self.R_unit, self.e_Vec[T], L, L, self.k)\
         for L in self.l])).flatten() for T in range(self.n)] )
        self.omega_mkMat = (omega_mk_Vec).T # make a omega_mk time seires

        # non-evolution
        # omega_mk_Vec = np.array( [(np.array([wmk(self.atilde, self.nu, self.p_Vec[0]/ self.R_unit, self.e_Vec[0], L, L, self.k)\
        #  for L in self.l])).flatten() ] * self.n ).reshape(-1, self.n)
        # self.omega_mkMat = (omega_mk_Vec) # make a omega_mk time seires


        timeb = time.time()
        # print(self.omega_mkMat)
        print("omega_mk finished, cost%6.3fs"%(timeb - timea))

        timea = time.time()
        hlmk_Vec = []
        calStep = 10000 # call harmonic every calStep step
        for i in range(self.n):
            if ((i % calStep) == 0 ):    
                ans = np.array([harmonic(self.atilde, self.nu, self.p_Vec[i]/ self.R_unit, self.e_Vec[i], l, l, self.k) for l in self.l])
                ans = np.vstack([item for item in ans])
                hlmk_cal = ans[:, 0:1]
                hlmk_Vec.append(hlmk_cal)
            else:
                hlmk = hlmk_cal
                hlmk_Vec.append(hlmk)     
        self.hlmkMat = np.concatenate(hlmk_Vec, axis = 1)
        timeb = time.time()
        print("hlmk finished, cost%6.3fs"%(timeb - timea))

        # See Ref.2 III (1)

        self.cn = len(self.omega_mkMat)   #column m
        self.factor = 1#(self.D / self.R_unit) 恢复公有制下大小
        self.phiMat = np.cumsum(self.omega_mkMat * (self.delta_t/self.t_unit), axis = 1)
   
        wave = (2.0 / self.factor ) *  self.hlmkMat * np.e**(-1j * self.phiMat) #evolve  Delta phi = int_ w dt
        wave = np.sum(wave, axis = 0)

        # See Ref.1 (48)
      
        self.hplus = wave.real
        self.hcross  = - wave.imag


    def eadd_dispersion(self, lambda_g, kbase=4): 
        self.lambda_g = lambda_g   
        self.kbase = kbase

        f_mkMat = self.omega_mkMat  / (2 * np.pi)
        f_baseVec = f_mkMat[int(self.kbase - min(self.k) ), :]

        timedelayMat = np.array([(1 + self.Z) * (self.D_a / self.R_unit) / \
            (2 * (self.lambda_g / self.R_unit)**2) * (1 / f_mkVeci**2 - 1 / f_baseVec**2 )\
             for f_mkVeci in f_mkMat]) 
        #G = c = 1

        self.phiMatD = self.phiMat + self.omega_mkMat * timedelayMat 
   

        # # See Ref.3 III (14)
        wave = (2.0 / self.factor ) *  self.hlmkMat * np.e**(-1j * self.phiMatD)
        wave = np.sum(wave, axis = 0)
        # # # See Ref.1 (48)

        self.hplus = wave.real# wave.real
        self.hcross  = - wave.imag




if __name__ == '__main__':

#   Parameters


    atilde = 0.9  # 自旋参量
    nu = 1e-05 # 质量比
    M = (1e6 * const.M_sun).to(u.kg).value  #the mass of system, kilogram
    # m = (1e1 * const.M_sun).to(u.kg).value  #the mass of small compact body, kilogram

    l = np.arange(2,3)      
    m = 2
    k = np.arange(-2,12,1)                   #the nth harmonic


    ########################
    R_unit = (G * M / C**2)#                #the unit length in G = C = 1 system, metre
    t_unit = R_unit / C    #                #the unit time in G = C = 1 system, second
    print(R_unit, t_unit)
    ########################

    e = 0.5                                 #eccentricity of orbit
    p = 12 * R_unit                         #semi-latus rectum

    a = p / (1 - e**2)                      #the semi-major axis of orbit, metre

    D = (1.00 * u.Gpc).to('m').value

#   Time Vector                  
    tb = 17542 *  t_unit  #17542 * t_unit                 #end time, second
    step = t_unit                           #step


    wave = teukolsky(atilde = atilde, nu = nu, M = M, l = l, k = k, e = e, a = a, D = D)
    wave.ecalculate(duration = tb, delta_t = step)

    h_plus_Vec = wave.hplus                  # h_plus of gravitational-wave
    h_cross_Vec = wave.hcross                # h_cross of gravitational-wave
    # print(h_plus_Vec)
    # assert 0>1

    ##h22 + h33###########################    
    l = np.arange(2,4)      
    wave2 = teukolsky(atilde = atilde, nu = nu, M = M, l = l, k = k, e = e, a = a, D = D)
    wave2.ecalculate(duration = tb, delta_t = step)
    h_plus_Vec2 = wave2.hplus                  # h_plus of gravitational-wave
    h_cross_Vec2 = wave2.hcross                # h_cross of gravitational-wave
    #####################################


    # print(h_cross_Vec)

#   dispersion
    lambda_g = (1.6e13 * u.km).to('m').value#the compton wavelength of graviton
    wave.eadd_dispersion(lambda_g)

    h_plus_d_Vec = wave.hplus                  # h_plus of gravitational-wave (dispersion)
    h_cross_d_Vec = wave.hcross                # h_cross of gravitational-wave (dispersion)


    a_Vec = wave.a_Vec           
    a_Vec = a_Vec / R_unit                     #G = C = 1 units

    e_Vec = wave.e_Vec
    p_Vec = a_Vec * (1 - wave.e_Vec**2)          #G = C = 1 units

    timeVec = wave.timeVec  
    vecLength = wave.n

    displayfactor = 8

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
    plt.plot(timeVec[0:int(vecLength/displayfactor)], h_plus_Vec[0:int(vecLength/displayfactor)],label = '$h_+(h22)$') 
    plt.plot(timeVec[0:int(vecLength/displayfactor)], h_plus_Vec2[0:int(vecLength/displayfactor)],label = '$h_+(h22 + h33)$') 

    plt.plot(timeVec[0:int(vecLength/displayfactor)], h_plus_d_Vec[0:int(vecLength/displayfactor)],label = '$hd_{+}$') 

    # plt.plot(timeVec[0:int(vecLength/8)], h_cross_Vec[0:int(vecLength/8)],label = '$h_x$')
    # plt.plot(timeVec[0:int(vecLength/8)], h_cross_d_Vec[0:int(vecLength/8)],label = '$hd_{x}$')
    plt.xlabel('t/kM')
    plt.xlabel('t/s')  
    plt.ylabel('h')
    plt.legend(loc ='best')
    # plt.show()
    plt.savefig("teukolsky.jpg", dpi = 150)    
    plt.close()


    output = np.array([timeVec, h_plus_Vec, h_cross_Vec, h_plus_d_Vec, h_cross_d_Vec,  e_Vec, p_Vec])
    output = output.T
    np.savetxt('teukolsky.out', output, delimiter=' ')   # X is an array

    output2 = np.concatenate([wave.hlmkMat, wave.omega_mkMat], axis = 0)
    output2 = output2.T
    np.savetxt('teukolsky_hlmk_omega.out', output2, delimiter=' ')   # X is an array





