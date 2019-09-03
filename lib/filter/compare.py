#!/usr/bin/env python
# -*- coding: utf-8 -*- 
"""
Created on Thu Mar 21 22:26:37 2019

Compute the overlap of two signals

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
import numpy as np  #import numpy, and name it as np
from scipy.interpolate import interp1d # interpolate 1-D
from lib.PSD.LISAcalPSD import LISAcalPSD1, LISAcalPSD2 



from pycbc import filter
from pycbc.types.timeseries import TimeSeries




class compare(object):
    """match class"""
    def __init__(self, data):
        self.data = np.array(data)
        self.result = None

    def template(self, para, func_template):
        self.paralen = len(para)
        self.func_template = np.frompyfunc(func_template,1,1)
        self.template = np.concatenate(self.func_template(para)).reshape(self.paralen, -1)  #func_template 应该是一个通用函数

    def calculate(self, func_matchway):
        self.result = np.array([func_matchway(self.data, template) for template in self.template])



def overlap_func(data, temp, psd, delta_t, f_min , f_max): 
    data = TimeSeries(data, delta_t=delta_t, copy=True) 
    temp = TimeSeries(temp, delta_t=delta_t, copy=True) 
    return filter.matchedfilter.overlap(data, temp, psd=psd, low_frequency_cutoff=f_min , high_frequency_cutoff=f_max, normalized=True)   

def match_func(data, temp, psd, delta_t, f_min , f_max): 
    data = TimeSeries(data, delta_t=delta_t, copy=True) 
    temp = TimeSeries(temp, delta_t=delta_t, copy=True) 

    amplitude1 = filter.matchedfilter.sigmasq(data, psd=psd, low_frequency_cutoff=f_min, high_frequency_cutoff=f_max )
    amplitude2 = filter.matchedfilter.sigmasq(temp, psd=psd, low_frequency_cutoff=f_min, high_frequency_cutoff=f_max )

    match,n =filter.matchedfilter.match(data, temp, psd=psd, low_frequency_cutoff=f_min , high_frequency_cutoff=f_max,v1_norm=True ,v2_norm=True)
    match = match/ np.sqrt(amplitude1 * amplitude2)
    return match


if __name__ == '__main__':
    main()


