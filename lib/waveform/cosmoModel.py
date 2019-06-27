#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Created on Sun Jun 23 17:00:00 2019

cosmos Model class

@author: Shucheng Yang
"""

#modules necessary
import numpy as np  #import numpy, and name it as np
np.set_printoptions(precision=17)

from astropy import constants as const #import constants 
import astropy.units as u              #import unit

from astropy.cosmology import FlatLambdaCDM  #宇宙学模型   
from astropy.cosmology import Planck13


from astropy.cosmology import z_at_value     #计算z的值
from scipy.integrate import quad                  #积分

# Constants
C = (const.c.value)

# functions
integrate = np.frompyfunc(quad, 3, 2)




zAtValue = lambda func, fval: z_at_value(func, fval, zmin = 1e-8, zmax = 1000, ztol = 1e-8, maxfun = 500) #设置z_at_value的参数,ztol最大1e-8 
zAtValue = np.frompyfunc(zAtValue, 2, 1) #将z _at_value 转为通用函数


def zAtValue_accu(func, fval, zmin=1e-16, zmax=1000, ztol=1e-16, maxfun = 5000):  
    count=0  

    while ( ((zmax-zmin)>ztol) and (count<=maxfun) ):  
        count+=1
        a = func(zmin) - fval
        b = func(zmax) - fval
        c = func((zmin + zmax)/2.0) - fval    
        if  a * c <0: 
            zmax= (zmin + zmax)/2.0
        elif b * c <0:
            zmin= (zmin + zmax)/2.0
        else:
            break
    return ((zmin + zmax)/2.0)

zAtValue_accu = np.frompyfunc(zAtValue_accu, 2, 1) # 转为通用函数

'''z_at_value, zAtValue_accu 
Parameters: 
func : function or method
A function that takes a redshift as input.

fval : astropy.Quantity instance
The value of func(z).

zmin : float, optional
The lower search limit for z. Beware of divergences in some cosmological functions, such as distance moduli, at z=0 (default 1e-8).

zmax : float, optional
The upper search limit for z (default 1000).

ztol : float, optional
The relative error in z acceptable for convergence.

maxfun : int, optional
The maximum number of function evaluations allowed in the optimization routine (default 500).
'''

# Class

class cosmoModel(object):
    """Parameters:  

        H0 : float or Quantity
        Hubble constant at z = 0. If a float, must be in [km/sec/Mpc]

        Om0 : float
        Omega matter: density of non-relativistic matter in units of the critical density at z=0.

        Tcmb0 : float or scalar Quantity, optional
        Temperature of the CMB z=0. If a float, must be in [K]. Default: 0 [K]. Setting this to zero will turn off both photons and neutrinos (even massive ones).

        Neff : float, optional
        Effective number of Neutrino species. Default 3.04.

        m_nu : Quantity, optional
        Mass of each neutrino species. If this is a scalar Quantity, then all neutrino species are assumed to have that mass. Otherwise, the mass of each species. The actual number of neutrino species (and hence the number of elements of m_nu if it is not scalar) must be the floor of Neff. Typically this means you should provide three neutrino masses unless you are considering something like a sterile neutrino.

        Ob0 : float or None, optional
        Omega baryons: density of baryonic matter in units of the critical density at z=0. If this is set to None (the default), any computation that requires its value will raise an exception.

        name : str, optional
        Name for this cosmological object.
"""
    def __init__(self, cosmo = FlatLambdaCDM(H0=69.3, Om0=0.286, Tcmb0=0, name="WAP9(with Or0,Ok0 = 0)")):
        #默认为一个忽略辐射项和曲率项的Flat LambaCDM宇宙， 密度参数数据来自 WAP9
        #(from Hinshaw et al. 2013, ApJS, 208, 19, doi: 10.1088/0067-0049/208/2/19. Table 4 (WMAP9 + eCMB + BAO + H0, last column))
        self.cosmo = cosmo
        self.name = cosmo.name

        self.H0 = cosmo.H0.value
        self.Om0 = cosmo.Om0
        self.Ode0 = cosmo.Ode0
        self.Tcmb0 = cosmo.Tcmb0.value

    def info(self):      
        print("\nCosmos Model Name: {name}\nH0 = {0:f}\nTcmb0 = {1:f}\nOm0 = {2:f}\nOde0 = {3:f}\n".format(self.H0, self.Tcmb0, self.Om0, self.Ode0, name = self.name))

        
    def D(self, Z):
        #return luminosity distance D at redshift Z 
        return (self.cosmo.luminosity_distance(Z)).to('m').value # m

    def D_a(self, Z, a = 0):
        #return distance alpha D_a at Z 
        D_a = lambda z: ((1 + z)**(a - 2))/np.sqrt(self.Om0*(1+z)**3 + self.Ode0)
        ans, error = integrate(D_a, 0, Z)  
        return ((( C * (1 + Z)**(1-a)/self.H0 ) * ans / 1e3) * u.Mpc).to("m").value                     #Mpc

    def z(self, D):
        #return redshift Z at luminosity distance D
        return zAtValue(self.D, D)

    def z_accu(self, D):
    #return redshift Z at luminosity distance D
        return zAtValue_accu(self.D, D)




if __name__ == '__main__':

#   Universe Model
    universe = cosmoModel()
    # universe = cosmoModel(cosmo = Planck13)
    
#   Model information
    
    universe.info()
    
    Z_array = np.array([0.20143642132540895, 0.2])
    D_array = universe.D(Z_array)
    D_a_array = universe.D_a(Z_array)

    print("\nZ_array = ", Z_array)
    print("D_array = ", D_array)
    print("D_a_array = ", D_a_array)


    Z_inverse_D_array = universe.z(D_array)
    Z_inverse_D_array_accu = universe.z_accu(D_array)

    print("\nZ_inverse_D_array = ", Z_inverse_D_array)
    print("Z_inverse_D_array_accu = ", Z_inverse_D_array_accu)





