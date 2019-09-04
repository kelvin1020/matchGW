#!/usr/bin/env python
# -*- coding: utf-8 -*- 
"""
Created on Thu Jul 13 10:25:00 2019
Generate wmk

@author: ysc
"""

import sys
import os 
##设置环境变量####################################################################################################################################################################################################################################################################################
path_exe = ((sys.path[0]).split(os.sep))#导入当前目录                
path_program = ((os.sep).join(path_exe)) 
path_program = path_program + '/teukolsky_func'
sys.path.append(path_program)           #将程序文件夹添加到环境变量
path_now = sys.path[0]                  #当前目录
##设置环境变量####################################################################################################################################################################################################################################################################################

import numpy as np
import time


def wmk_uniarg(*args):
    'wmk'
    # np.set_printoptions(precision=18)
    atilde, nu, p, e, l, m, k = args[0]
 
    wr = np.zeros(1, dtype = np.float64)   #该列表的元素会被操作
    wp = np.zeros(1, dtype = np.float64)
   
    etwave.eoborbit(atilde, nu, p, e, wr, wp)

    return m * wp[0] + k * wr[0]





def wmk(atilde, nu, p, e, l, m, k):
    'wmk'
    # np.set_printoptions(precision=18)
 
    wr = np.zeros(1, dtype = np.float64)   #该列表的元素会被操作
    wp = np.zeros(1, dtype = np.float64)
   
    etwave.eoborbit(atilde, nu, p, e, wr, wp)

    return m * wp[0] + k * wr[0]



if __name__ != '__main__':
    import lib.waveform.teukolsky_func.etwave as etwave

if __name__ == '__main__':

    import etwave
    np.set_printoptions(precision=18)
#   Parameters
    atilde = 0.9e0  # s自旋参量
    nu = 1e-03 # 质量比
    p = 12e0   #半通徑
    e = 0.5e0  #偏心率
    l = 2.0
    m = l


    k = np.arange(-2, 12)
    print(k)
  

    ta = time.time()   
    ans = wmk(atilde, nu, p, e, l, m, k)
    tb = time.time()
    print("用时%6.3fs"%(tb-ta))
    print(ans)


    arg = [atilde, nu, p, e, l, m, k]

    ta = time.time()   
    ans = wmk_uniarg(arg)
    tb = time.time()
    print("用时%6.3fs"%(tb-ta))
    print(ans)
