#!/usr/bin/env python
# -*- coding: utf-8 -*- 
"""
Created on Sun Mar 17 20:00:00 2019

Calculate the orbit coefficient e,a of EMRI or IMRI(high accuracy)

@author: Shucheng Yang
"""

#modules necessary
import numpy as np  #import numpy, and name it as np
from astropy import constants as const #import constants 
import astropy.units as u              #import unit

from scipy.integrate import odeint     #import ordinary differential equation integrator

#modules used for test
import matplotlib.pyplot as plt        #import pyplot to plot


def dedv_dt8(atilde, rat, ec, vm):
    """ 
        intermeidate function that return the da_dt, de_dt  

        Shucheng Yang, Mar 2019

        Parameters
        ----------
        atilde: the deformed Kerr spin parameter
        rat : sysmetry mass ratio
        ec : eccentricity of orbit
        vm : sqrt(1/pm), where pm is semi-latus rectum, G = c = 1 

        Returns
        -------
        dv_dt : the derivative of v(t)
        de_dt : the derivative of e(t)

        References
        ----------
        1. Sago, N. & Fujita, R. Calculation of radiation reaction effect on orbital parameters in Kerr spacetime. Prog. Theor. Exp. Phys. 2015, 073E03 (2015).

        p.26: Appendix C. Secular evolution of the orbital parameters v, e, and Y 
    """
    gamma = 0.57721566490153286060651209e0 # Eular gamma
    cosuta = 1e0 # cos(uta) = Lz/sqrt(Lz^2+Q), on equatotial plane = 1


    dvdt_N = (32e0/5)*rat*vm**9*(-ec**2+1e0)**(3e0/2e0)

    dedt_N = (-304e0/15)*rat*vm**8*ec*(-ec**2+1e0)**(3e0/2e0)


    dvdt8_c = 1+7e0/8*ec**2+(-743e0/336-55e0/21*ec**2+8539e0/\
          2688*ec**4)*vm**2+(-133e0/12*cosuta*atilde+4*\
          np.pi+(-379e0/24*cosuta*atilde+97e0/8*np.pi)*ec**2+(-475e0/96*\
          cosuta*atilde+49e0/32*np.pi)*ec**4-49e0/4608*np.pi*ec**6)*\
          vm**3+(815e0/96*atilde**2*cosuta**2-329e0/96*atilde**2+\
          34103e0/18144+(-929e0/96*atilde**2-526955e0/\
          12096+477e0/32*atilde**2*cosuta**2)*ec**2+(-1232809e0/\
          48384+999e0/256*atilde**2*cosuta**2-\
          1051e0/768*atilde**2)*ec**4+105925e0/\
          16128*ec**6)*vm**4+(-1451e0/56*cosuta*atilde-4159e0/\
          672*np.pi+(-1043e0/96*cosuta*atilde-\
          48809e0/1344*np.pi)*ec**2+(-15623e0/336*\
          cosuta*atilde+679957e0/43008*np.pi)*ec**4+(-35569e0/1792*\
          cosuta*atilde+\
          4005097e0/774144*np.pi)*ec**6)*vm**5+(-1712e0/105*\
          np.log(vm)+16e0/3*np.pi**2-3424e0/105*np.log(2e0)-289e0/6*\
          atilde*np.pi*cosuta+\
          145759e0/1344*atilde**2*cosuta**2-1712e0/105*gamma+\
          16447322263e0/139708800-331e0/192*atilde**2+\
          (-24503e0/210*np.log(vm)-4225e0/24*atilde*\
          np.pi*cosuta+1391e0/30*np.log(2e0)-24503e0/210*gamma-\
          78003e0/280*np.log(3e0)+229e0/6*np.pi**2+\
          27191e0/224*atilde**2*cosuta**2+8901670423e0/\
          11642400+2129e0/42*atilde**2)*ec**2+(-56239e0/10752*atilde**2+\
          3042117e0/1120*np.log(3e0)-11663e0/140*gamma+\
          269418340489e0/372556800-418049e0/84*np.log(2e0)-\
          11663e0/140*np.log(vm)-17113e0/192*atilde*np.pi*\
          cosuta+414439e0/3584*atilde**2*cosuta**2+109e0/4*np.pi**2)*\
          ec**4+(23e0/16*np.pi**2-\
          1044921875e0/96768*np.log(5e0)-3571e0/\
          3584*atilde**2-2461e0/560*np.log(vm)+41071e0/1536*atilde**2*\
          cosuta**2-108577e0/13824*atilde*np.pi*cosuta-2461e0/560*gamma-\
          42667641e0/3584*np.log(3e0)+174289281e0/862400+\
          94138279e0/2160*\
          np.log(2e0))*ec**6)*vm**6+(1013e0/16*cosuta*\
          atilde**3-809e0/48*atilde**2*np.pi-3443e0/24*cosuta**3*atilde**3+\
          1775e0/48*np.pi*atilde**2*\
          cosuta**2-358901e0/2016*cosuta*atilde-4415e0/4032*np.pi+\
          (-27671e0/96*cosuta**3*atilde**3+15395e0/96*np.pi*atilde**2*\
          cosuta**2+5917e0/32*cosuta*atilde**3-\
          3487e0/32*atilde**2*np.pi+710033e0/2016*cosuta*\
          atilde-2647367e0/8064*np.pi)*ec**2+(-2937e0/16*cosuta**3*atilde**3+\
          141973e0/1536*np.pi*atilde**2*cosuta**2+45775e0/\
          384*cosuta*atilde**3-35121e0/512*atilde**2*np.pi+4780219e0/\
          16128*cosuta*atilde-363789061e0/774144*np.pi)*ec**4+\
          (-15863e0/768*cosuta**3*atilde**3+\
          95233e0/13824*np.pi*atilde**2*cosuta**2+6161e0/\
          768*cosuta*atilde**3-41671e0/13824*atilde**2*np.pi-1267799e0/\
          96768*cosuta*atilde-394967495e0/41803776*np.pi)*ec**6)*vm**7\
         +\
        (-44353e0/336*atilde*np.pi*cosuta+124741e0/4410*np.log(vm)\
         +124741e0/4410*gamma-47385e0/1568*np.log(3e0)+3959271176713e0/25427001600+10703e0/768*atilde**4-\
         19223e0/384*cosuta**2*atilde**4-2418889e0/20736*atilde**2+134871689e0/145152*atilde**2*cosuta**2+40223e0/768*cosuta**4*atilde**4\
          -361e0/126*np.pi**2+127751e0/1470*np.log(2e0)+(-15595889e0/8820*np.log(2e0)-47049113e0/145152*atilde**2-\
         23185e0/252*np.pi**2+4632623e0/8820*gamma-100313e0/384*cosuta**2*atilde**4-32545e0/168*atilde*np.pi*cosuta+64499905\
         /48384*atilde**2*cosuta**2+1518507e0/784*np.log(3e0)+74939e0/768*atilde**4+4632623e0/8820*np.log(vm)+45437e0/256*cosuta**\
         4*atilde**4-24358472380577e0/33902668800)*ec**2+(548049031e0/8820*np.log(2e0)+829453e0/6144*cosuta**4*atilde**\
         4-92184131e0/387072*atilde**2-3173828125e0/301056*np.log(5e0)+6657731e0/8820*np.log(vm)-11824457e0/\
         21504*atilde*np.pi*cosuta-10400134887e0/501760*np.log(3e0)+71789387e0/55296*atilde**2*cosuta**2-197599e0/1024*cosuta**2*atilde\
         **4+137903e0/2048*atilde**4-95375799914137e0/40683202560-28789e0/252*np.pi**2+6657731e0/8820*\
         gamma)*ec**4+(3545e0/256*cosuta**4*atilde**4+48911531e0/96768*atilde**2*cosuta**2+5515e0/1536*atilde**4-\
         302877469777873e0/203416012800-67267451e0/193536*atilde*np.pi*cosuta-462761e0/3456*atilde**2+225243e0/\
         7840*np.log(vm)+59531e0/2016*np.pi**2-62212514083e0/90720*np.log(2e0)+225243e0/7840*gamma+\
          3788091765e0/100352*np.log(3e0)+314655859375e0/1161216*np.log(5e0)-8071e0/512*cosuta**2*atilde**4)*ec**6)*vm**8\

    dedt8_c= 1e0+121e0/304*ec**2+(-6849e0/2128-2325e0/2128*\
          ec**2+22579e0/17024*\
          ec**4)*vm**2+(-879e0/76*cosuta*atilde+985e0/152*np.pi+\
          (-699e0/76*cosuta*atilde+5969e0/608*np.pi)*ec**2+(-1313e0/\
          608*cosuta*\
          atilde+24217e0/29184*np.pi)*ec**4)*vm**3+(-286397e0/38304+\
          5869e0/608*\
          atilde**2*cosuta**2-3179e0/608*atilde**2+(633e0/64*atilde**2*\
          cosuta**2-8925e0/1216*atilde**2-\
          2070667e0/51072)*ec**2+(-3191e0/4864*atilde**2-\
          3506201e0/306432+9009e0/4864*atilde**2*cosuta**2)*\
          ec**4)*vm**4+(-1903e0/304*cosuta*atilde-87947e0/4256*\
          np.pi+(-3539537e0/68096*np.pi-93931e0/8512*cosuta*atilde)*\
          ec**2+(-442811e0/17024*cosuta*atilde+5678971e0/817152*\
          np.pi)*ec**4)*vm**5+(11224646611e0/46569600-\
          234009e0/5320*np.log(3e0)+180255e0/8512*atilde**2+\
          598987e0/8512*atilde**2*\
          cosuta**2-11021e0/285*np.log(2e0)-82283e0/\
          1995*gamma-82283e0/1995*np.log(vm)-11809e0/152*atilde*np.pi*\
          cosuta+769e0/57*np.pi**2+(-2982946e0/1995*np.log(2e0)+\
          536653e0/8512*atilde**2+356845e0/8512*atilde**2*cosuta**2-\
          297674e0/1995*np.log(vm)-297674e0/1995*gamma+\
          927800711807e0/884822400+1638063e0/3040*np.log(3e0)+\
          2782e0/57*np.pi**2-91375e0/608*atilde*np.pi*cosuta)*ec**2+\
          (-1044921875e0/204288*np.log(5e0)+190310746553e0/\
          262169600+760314287e0/47880*np.log(2e0)-1739605e0/\
          29184*atilde*np.pi*cosuta-1147147e0/15960*gamma-1022385321e0/\
          340480*np.log(3e0)-1147147e0/15960*np.log(vm)+10721e0/\
          456*np.pi**2+56509e0/9728*atilde**2+3248951e0/68096*atilde**2*\
          cosuta**2)*ec**4)*vm**6\
           +\
         (2601e0/38*cosuta*atilde**3+20079e0/304*np.pi*atilde**2*cosuta**2+320719e0/2736*cosuta*atilde-20151e0/152*cosuta**3*atilde**3-4988783e0/153216*np.pi-5561e0/\
         152*atilde**2*np.pi+(-509006417e0/1225728*np.pi-291031e0/2432*atilde**2*np.pi-59529e0/304*cosuta**3*atilde**3+9760833\
          /17024*cosuta*atilde+88917e0/608*cosuta*atilde**3+20449e0/128*np.pi*atilde**2*cosuta**2)*ec**2+(2363e0/38*cosuta*atilde**3-993385e0/\
          19456*atilde**2*np.pi-19112540225e0/44126208*np.pi-112313e0/1216*cosuta**3*atilde**3+43942891e0/153216*cosuta*atilde+\
         2053643e0/29184*np.pi*atilde**2*cosuta**2)*ec**4)*vm**7+(20995469e0/48384*atilde**2*cosuta**2-5673e0/133*np.pi**2-9392e0/\
         665*np.log(2e0)+43300413e0/148960*np.log(3e0)+874376e0/4655*np.log(vm)+150603e0/4864*atilde**4+874376e0/4655*\
         gamma-33669848060399e0/53679225600-361423e0/4256*atilde*np.pi*cosuta-195167e0/2432*cosuta**2*atilde**4-\
         94271041e0/919296*atilde**2+290707e0/4864*cosuta**4*atilde**4+(234596963e0/262656*atilde**2*cosuta**2+1557117e0/9728\
         *cosuta**4*atilde**4+3625715e0/68096*atilde*np.pi*cosuta-759186683e0/1838592*atilde**2+1048805e0/9728*atilde**4-1284325e0/\
         4864*cosuta**2*atilde**4+11254601e0/9310*np.log(vm)+11254601e0/9310*gamma-5446366938713179e0/\
         1288301414400+444813059e0/27930*np.log(2e0)-1349385561e0/340480*np.log(3e0)-36794e0/133*np.pi**2-\
         3173828125e0/1430016*np.log(5e0))*ec**2+(1860004949e0/2451456*atilde**2*cosuta**2-410971e0/2128*np.pi**2+\
         74356697e0/74480*gamma-228821185e0/817152*atilde*np.pi*cosuta+15152854761e0/9533440*np.log(3e0)+1922507\
         /38912*atilde**4-20399023129e0/95760*np.log(2e0)-2524399e0/19456*cosuta**2*atilde**4-7196522540829509e0/\
         1288301414400+1618059765625e0/17160192*np.log(5e0)+3282195e0/38912*cosuta**4*atilde**4-1927502009e0/\
         7354368*atilde**2+74356697e0/74480*np.log(vm))*ec**4)*vm**8;


    dvdtH_c= (-1e0/4*cosuta*atilde-9e0/32*cosuta*atilde**3-15e0/32*\
          cosuta**3*atilde**3+\
          (-27e0/32*cosuta*atilde**3-3e0/4*cosuta*atilde-45e0/32*\
          cosuta**3*atilde**3)*ec**2+\
          (-3e0/32*cosuta*atilde-45e0/256*cosuta**3*atilde**3-27e0/256*\
          cosuta*atilde**3)*ec**4)*vm**5+(-15e0/64*cosuta**3*atilde**3-\
          189e0/64*cosuta*atilde**3-11e0/8*cosuta*atilde+(45e0/32*\
          cosuta**3*atilde**3-81e0/4*cosuta*atilde**3-\
          69e0/8*cosuta*atilde)*ec**2+(-7479e0/512*cosuta*\
          atilde**3-381e0/64*cosuta*atilde+1035e0/512*cosuta**3*atilde**3)*\
          ec**4+(45e0/512*cosuta**3*atilde**3-\
          423e0/512*cosuta*atilde**3-11e0/32*\
          cosuta*atilde)*ec**6)*vm**7 

    dedtH_c= (-33e0/608*(15*atilde**2*cosuta**2+9*atilde**2+8)*cosuta*atilde-99e0/\
          1216*(15*atilde**2*cosuta**2+9*atilde**2+8)*cosuta*atilde*ec**2-33e0/\
          4864*(15*atilde**2*cosuta**2+9*atilde**2+8)*cosuta*atilde*ec**4)*\
          vm**5+(3e0/9728*cosuta*atilde*(-120*atilde**2*cosuta**2-\
          21672*atilde**2-9664)+3e0/9728*cosuta*atilde*(7920*atilde**2*cosuta**2-\
          76248*atilde**2-31776)*ec**2+3e0/9728*cosuta*atilde*(5505*atilde**2*\
          cosuta**2-37197*atilde**2-15064)*ec**4)*vm**7

    dv_dt = dvdt_N*(dvdt8_c+dvdtH_c)
    de_dt = dedt_N*(dedt8_c+dedtH_c)

    return de_dt, dv_dt



#ordinary differential equation integrator
def evolve(atilde, rat, ec, vm, timeVec):
    """ 
        ODE integrator that return the e Vector and a Vector

        Shucheng Yang, Mar 2019

        Parameters
        ----------
        atilde: the deformed Kerr spin parameter
        rat : sysmetry mass ratio
        ec : eccentricity of orbit
        vm : sqrt(1/pm), where pm is semi-latus rectum, G = c = 1 
        timeVec : array_like, the time sequence

        Returns
        -------
        e_Vec : array_like, the e sequence
        v_Vec : array_like, the v sequence

    """

    def odeg(vrb, t):#return de_dt, dv_dt every step
        ec, vm = vrb.tolist()# a list that store the e_Vec and v Vec
        de_dt, dv_dt = dedv_dt8(atilde, rat, ec, vm)
        return de_dt,dv_dt 

  
    init_status = ec, vm  #initial value
    args = () #Constants
    result = odeint(odeg, init_status, timeVec, args)# e_Vec and a Vec's list(n by 2 array)

    e_Vec = result[:,0] 
    v_Vec = result[:,1] 

    return e_Vec, v_Vec


if __name__ == '__main__':

    #   Constants

    G = (const.G).value # gravitational constant
    C = (const.c).value # the speed of light

    #   Parameters
    M_ = 1e6
    M = (M_ * const.M_sun).to(u.kg).value  #the mass of system, kilogram
    rat = 1e-4
    m = 1/2 * M * ( 1- np.sqrt(1- 4* rat))  #the mass of small compact body, kilogram
    m_ = (m / const.M_sun).value
    ########################
    R_unit = (G * (M) / C**2)#                #the unit length in G = C = 1 system, metre
    t_unit = R_unit / C    #                #the unit time in G = C = 1 system, second
    ########################
    print(t_unit)

    e = 0.7                                 #eccentricity of orbit
    pm = 12
    am = pm/(1 - e**2)
    print("e ={}, p = {}, a = {}, rp={}".format(e,pm,am, pm/(1+e)))
    print("R_unit ={}, t_unit = {}".format(R_unit, t_unit))

    a = am * R_unit                         #the semi-major axis of orbit, metre
    p = a * (1 - e**2)

    #   Time Vector
    ta = 0                                  #start time, G = C = 1 units
    tb = 2**24#6.035e6# *  t_unit               #end time
    step = 2**2#tb /2**21#2**2#t_unit                           #step

    timeVec = np.arange(ta, tb, step)       #time Vector(time sequence)


    ########################haccu########################

    atilde = 0.9
    vm = np.sqrt(1/pm)

    tbm = tb/t_unit
    stepm = step/ t_unit
    timeVecm = np.arange(ta, tbm, stepm)       #time Vector(time sequence)

    z1 = 1 + (1 -atilde**2)**(1/3) * ((1+atilde)**(1/3) + (1-atilde)**(1/3))
    z2 = np.sqrt(3 * atilde**2 + z1**2)
    r_isco = 3 + z2 - np.sqrt((3 - z1) * (3 + z1 + 2* z2)) # prograde case, 自旋与轨道角动量同向
    print("M= {:6.2e}, rat = {}, m ={}, atilde={}, r_isco={}".format(M_, rat,m_, atilde, r_isco))
    ########################haccu########################
    evolve_hacc8 = evolve

    e_Vec, vm_Vec =  evolve_hacc8(atilde, rat, e, vm, timeVecm) #e Vector and a Vector
    pm_Vec =  1 / vm_Vec**2                    #G = C = 1 units
    am_Vec =  pm_Vec / (1 - e_Vec**2)
    rp_Vec = am_Vec * (1 - e_Vec)            #近心点


    #graph
    timeVeckm = timeVec / (1000 * t_unit)        #G = C = 1 units, kM
    plt.figure(figsize=(10,8))
    #plot t and p(rp)
    plt.subplot(2,2,1)  
    plt.plot(timeVeckm, pm_Vec, label = 'haccu8-pm')  
    plt.plot(timeVeckm, rp_Vec, label = 'haccu8-rp')
    plt.plot(timeVeckm, np.ones(len(timeVeckm)) * r_isco, label = 'isco')
    plt.ylim([0, 12])
    plt.xlabel('t/kM')
    plt.ylabel('p/M')
    plt.legend()

    #plot e and p
    plt.subplot(2,2,2)    
    plt.plot(e_Vec, pm_Vec, label = 'haccu8-pm')
    plt.gca().invert_xaxis()
    plt.ylim([0, 12])
    plt.title("e and p")
    plt.xlabel('e')
    plt.ylabel('p/M')
    plt.legend()

    #plot t and e
    plt.subplot(2,2,4) 
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.plot( e_Vec, timeVeckm, label = 'haccu8')
    plt.xlabel('e')
    plt.ylabel('t/kM')
    plt.legend()
    plt.show()


