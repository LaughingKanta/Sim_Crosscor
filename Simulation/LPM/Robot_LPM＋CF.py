#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 19:08:30 2021

LPM + CF simulation
@author: hasegawakanta
"""

def Initialize():
    #sensing_setting
    Fs=1000000 #[Hz]sampling_f
    FFT_Smpl=2**15 #Number_of_samplings_for_sensing[smpls]
    
    #pulse_design_setting
    Amplitude = 32767. / 2.   #pulse Amplitude: 
    Fi = 80000    #: initial frequency [Hz]
    Ft = 40000  #: terminal frequency [Hz]
    Duration =0.005 #LPM_duration [s]
    Duration_CF =0.005 # CF_duration [s]
    
    draw_tx_spc_flg=True
    draw_Acor_flg=True
    draw_tx_FFT_fl=True
    
    return Fs,FFT_Smpl,Amplitude,Fi,Ft,Duration,Duration_CF,draw_tx_spc_flg,draw_Acor_flg,draw_tx_FFT_fl

