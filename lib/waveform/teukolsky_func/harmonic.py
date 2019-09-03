#!/usr/bin/env python
# -*- coding: utf-8 -*- 
"""
Created on Thu Jul 13 10:25:00 2019
Generate evolved waveform of gravitational-wave(without dispersion)

@author: ysc
"""


import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
import time

from functools import partial

def _harmonic(atilde, nu, p, e, l, m, k):
    'teukolsky harmonic'
    # np.set_printoptions(precision=18)
    hlmk = np.zeros(1, dtype = np.complex128) #该列表的元素会被操作
    wr = np.zeros(1, dtype = np.float64)
    wp = np.zeros(1, dtype = np.float64)
    etwave.etwavefrom(atilde, nu, p, e, l, m, k, hlmk, wr, wp)

    return hlmk[0], m * wp[0] + k * wr[0]


def harmonic(atilde, nu, p, e, l, m, k):
    func = partial(_harmonic, atilde, nu, p, e, l, m) # 提取x作为partial函数的输入变量
    with Pool() as pool:
        return pool.map(func, k)







def _wave_harmonic(atilde, nu, p, e, l, k, t=0):
    'teukolsky wave(t),l = m,k (element)'
    np.set_printoptions(precision=18)
    hlmk = np.zeros(1, dtype = np.complex128) #该列表的元素会被操作
    wr = np.zeros(1, dtype = np.float64)
    wp = np.zeros(1, dtype = np.float64)
    etwave.etwavefrom(atilde, nu, p, e, l, l, k, hlmk, wr, wp)
    return 2 * hlmk[0] * np.e**(-1j * (l * wp[0] + k * wr[0]) * t)#l = m

def uwave_harmonic(atilde, nu, p, e, l, k, t):
    'teukolsky wave(t),l = m,k (list)'
    func = partial(_wave_harmonic, atilde, nu, p, e, l, t=t) # 提取x作为partial函数的输入变量
    with Pool() as pool:

        return np.sum(pool.map(func, k))


if __name__ != '__main__':
    import lib.waveform.teukolsky_func.etwave as etwave

if __name__ == '__main__':
    import etwave

    # np.set_printoptions(precision=18)
#   Parameters
    atilde = 0.9e0  # 自旋参量
    nu = 1e-03 # 质量比
    p = 12e0   #半通徑
    e = 0.5e0  #偏心率
    l = 2.0
    m = l

    core = cpu_count()
    print("核心数 = %d"%core)
    
    k = np.arange(-2, 12)
    print(k)
  

    t = 0

    #此程序有个很奇怪的bug, 在并行uwave_harmonic函数 前执行任何 etwavefrom 的显式调用（或执行harmonic, wave_harmonic）
    #都会造成计算结果出错。而单独用 etwaveform 则也不会出错

    # func = partial(_wave_harmonic, atilde, nu, p, e, l, t=t) # 提取x作为partial函数的输入变量
    # ta = time.time()    
    # ans = np.sum(list(map(func, k))) #串行
    # tb = time.time()
    # print("串行用时%6.3f"%(tb-ta))
    # print(ans)

    ta = time.time()   
    ans = uwave_harmonic(atilde, nu, p, e, l, k, t)
    tb = time.time()
    print("并行用时%6.3fs"%(tb-ta))
    print(ans)

