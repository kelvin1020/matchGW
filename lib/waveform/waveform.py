#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Created on Thu Mar 21 22:26:37 2019

Compute the overlap of two signals

@author: ysc
"""

##设置环境变量####################################################################################################################################################################################################################################################################################
import sys
import os 
path_exe = ((sys.path[0]).split(os.sep))#导入当前目录
path_exe.pop()                                  
path_program = ((os.sep).join(path_exe)) 
sys.path.append(path_program)           #将程序文件夹添加到环境变量
path_now = sys.path[0]                  #当前目录
######################################################################################################################################################################################################################################################################################


import numpy as np
from pycbc.types.timeseries import TimeSeries


# The output of this function are the "plus" and "cross" polarizations of the gravitational-wave signal 
# as viewed from the line of sight at a given source inclination (assumed face-on if not provided)


class waveform_template(object):
	"""docstring for waveform_template"""
	def __init__(self):
		self.name = None

		
def get_td_waveform(template=None, **kwargs):

	"""Return the plus and cross polarizations of a time domain waveform.

	Parameters
	----------
	template: object
		An object that has attached properties. This can be used to subsitute
		for keyword arguments. A common example would be a row in an xml table.
	{params}

	Returns
	-------
	hplus: TimeSeries
		The plus polarization of the waveform.
	hcross: TimeSeries
		The cross polarization of the waveform.
	"""
	hplus = None
	hcross = None

	if (not template):
		initial_array = np.ones(100)
		hplus = TimeSeries(initial_array, delta_t=kwargs['delta_t'], epoch='', dtype=None, copy=True)
		hcross = TimeSeries(initial_array, delta_t=kwargs['delta_t'], epoch='', dtype=None, copy=True)
	else:

		hplus = TimeSeries(template.hplus, delta_t=template.delta_t, epoch='', dtype=None, copy=True)
		hcross = TimeSeries(template.hcross, delta_t=template.delta_t, epoch='', dtype=None, copy=True)

	return hplus, hcross


if __name__ == '__main__':

	print("test function get_td_waveform(without template)")
	sampIntrvl = 0.5
	hp, hc = get_td_waveform(delta_t = sampIntrvl)
	print(hp,hc)


	print("\ntest function get_td_waveform(with template)")

	from astropy import constants as const #import constants 
	import astropy.units as u              #import unit
	from waveform import *
	from peters import *
	from cosmoModel import *
	G = (const.G).value # gravitational constant
	C = (const.c).value # the speed of light

	#peters
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

	duration = 2**16
	sampIntrvl = 4
	lambda_g = (1.6e13 * u.km).to('m').value#the compton wavelength of graviton

	wave = peters(n = n, M = M, m = m, e = e, a = a, D = D)
	wave.calculate(duration = duration, delta_t = sampIntrvl)
	wave.add_dispersion(lambda_g)



	hp, hc = get_td_waveform(template=wave)
	print(hp,hc)
