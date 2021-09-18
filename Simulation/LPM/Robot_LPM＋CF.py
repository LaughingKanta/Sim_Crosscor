#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 19:08:30 2021

LPM + CF simulation
@author: hasegawakanta
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.signal import hilbert
import scipy.signal as signal
import pathlib


def Initialize():
    #sensing_setting
    Fs=1000000 #[Hz]sampling_f
    FFT_Smpl=2**15 #Number_of_samplings_for_sensing[smpls]
    
    #pulse_design_setting
    Amplitude = 32767. / 2.   #pulse Amplitude: 
    Fi = 80000    #: initial frequency [Hz]
    Ft = 40000  #: terminal frequency [Hz]
    Duration =0.0075 #LPM_duration [s]
    Duration_CF =0.0025 # CF_duration [s]
    Duration_space=0.0001
    
    draw_tx_spc_flg=True
    draw_Acor_flg=True
    draw_tx_FFT_flg=True
    print("ok!")
    
    return Fs,FFT_Smpl,Amplitude,Fi,Ft,Duration,Duration_CF,Duration_space,draw_tx_spc_flg,draw_Acor_flg,draw_tx_FFT_flg
    
#位相を出している？
def LPM_model_h(fs, fi, ft, dur, t):
    #phai=alpha*ln(gamma*t+xi)+beta=2*np.pi*f*tから初期・終期のt,f条件から解析的に解いて代入した
    Ts=1.0/fs #[Hz]sampling_period
    k=(1-ft/fi)/(ft*dur)
    f=fi/(1+k*fi*t)
    del_phase = 2*np.pi*f*Ts
    return del_phase
#位相を出している？
def CF_add(fs, ft, t):
    Ts=1.0/fs #[Hz]sampling_period
    f=ft*t/t
    del_phase = 2*np.pi*f*Ts
    return del_phase
    
#波形を作った？　なぜ加算して行っているのか・・
def trans_wv(del_phase_array):
    phase=0
    wave=np.zeros(len(del_phase_array))
    
    for i in range(len(del_phase_array)):
        phase = phase +del_phase_array[i]
        wave[i] = np.sin(phase)
    return wave


def zero_filled(wv,t):
    wv=np.append(wv, np.zeros(len(t)-len(wv)))
    return wv

def Xcor_func(t, Txwv, Rxwv):
    txspc = fft(Txwv)    
    crwv = ifft(fft(Rxwv) * np.conj(txspc)).real
    hlbwv = hilbert(crwv)
    env = np.abs(hlbwv) 
    return env


def draw_wv(fig_name, x_data, y_data, x_min, x_max, y_min, y_max, f_init, f_termin, val):
    fig, (ax1) = plt.subplots(1,1, sharex=True, constrained_layout=True)
    ax1.plot(x_data, y_data)
    
    ax1.set_xlim(0, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_title(r"$f_i$ = {} kHz  $f_t$ = {} kHz".format(str(int(f_init/1000)), str(int(f_termin/1000))))
    ax1.set_ylabel("waveform")
    ax1.set_xlabel("Time [s]")
    fig.savefig(val/'{}_{}_{}.png'.format(fig_name, str(int(f_init/1000)), str(int(f_termin/1000))))
#    plt.show()
    fig.clear()
    ax1.clear()
    plt.close()
    
def draw_spctrm(fig_name, x_data, y_data, x_min, x_max, y_min, y_max, f_init, f_termin, val):
    fig, (ax1) = plt.subplots(1,1, sharex=True, constrained_layout=True)
    ax1.plot(x_data, y_data)
    
    ax1.set_xlim(0, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_title(r"$f_i$ = {} kHz  $f_t$ = {} kHz".format(str(int(f_init/1000)), str(int(f_termin/1000))))
    ax1.set_ylabel("Frequency [kHz]")
    ax1.set_xlabel("Amplitude [a.u.]")
    fig.savefig(val/'{}_{}_{}.png'.format(fig_name, str(int(f_init/1000)), str(int(f_termin/1000))))
    plt.show()
    fig.clear()
    ax1.clear()
    plt.close()

def draw_spectrogram(fig_name, x_data, y_data, z_color, f_init, f_termin, val):
    plt.rcParams["font.size"] = 20
    x_data=x_data*1000
    y_data=y_data/1000
    y_max=100
    f=plt.figure(figsize=(8.0, 8.0))
    plt.pcolormesh(x_data, y_data, z_color, cmap="GnBu")
    plt.title(r"$ f_i$ = {} kHz  $f_t$ = {} kHz".format(str(int(f_init/1000)), str(int(f_termin/1000))))
    plt.xlim([0,max(x_data)])
    plt.ylim([0,y_max])
    plt.xlabel("Time [ms]")
    plt.ylabel("Frequency [kHz]")
    plt.colorbar()
    plt.savefig(val/'{}_spctrgm{}_{}.png'.format(fig_name, str(int(f_init/1000)), str(int(f_termin/1000))))
    plt.show()
    f.clear()
    plt.close()
    

def tx_design_view(fs, FFT_smpl, Amp, fi, ft, duration, duration_CF,duration_space,draw_tx_spc_flg, draw_Acor_flg, draw_tx_FFT_flg):

    Ts=1.0/fs #[s]
    t_start = 0.0
    t_array = np.arange(t_start, FFT_smpl*Ts, Ts)
    
    ######################################
    ##保存場所
    dataDir = pathlib.Path('Output/AddSpace')
        
    ###### LPM_FM + CF_ver ######

    ###### transmit_phase(txphase_CF)_and_reference_phase(txphase)_calclation ######    
    n_dur=int(duration/Ts)+1
    n_dur_CF=int(duration_CF/Ts)+1
    n_dur_space=int(duration_space/Ts)+1

    phase_space=t_array[n_dur:n_dur+n_dur_space]
    txphase = LPM_model_h(fs, fi, ft, duration, t_array[:n_dur])
    txphase_space=np.zeros(len(phase_space))
    txphase_CF = CF_add(fs, ft, t_array[n_dur_space:n_dur_space+n_dur_CF])
    
    ##LPM と　CFの位相を結合
    txphase_space=np.hstack([txphase,txphase_space])
    txphase_CF=np.hstack([txphase_space,txphase_CF])
    ###### transmit_wave(txwv_CF)_and_reference_wave(txwv)_calclation######    
    txwv = trans_wv(txphase)
    txwv = zero_filled(txwv, t_array)
    txwv_CF = trans_wv(txphase_CF)
    txwv_CF = zero_filled(txwv_CF, t_array)

    ##### tx_wave_draw ###############
#    draw_wv("txwv", t_array, txwv, min(t_array), max(t_array), min(txwv), max(txwv), fi, ft, dataDir)#生波形描画用
        
    ##### tx_spectrogram_draw ###############
    ###### transmit_wave("tx_CF")_and_reference_wave("tx")_draw######    
    if draw_tx_spc_flg==1:
        print(fi, ft)
        spc_f, spc_t, spc_pw = signal.spectrogram(txwv, fs, nperseg=1024, noverlap=int(1024*0.98))
        draw_spectrogram("tx", spc_t, spc_f, spc_pw, fi, ft, dataDir)   
        spc_f_CF, spc_t_CF, spc_pw_CF = signal.spectrogram(txwv_CF, fs, nperseg=1024, noverlap=int(1024*0.98))
        draw_spectrogram("tx_CF", spc_t_CF, spc_f_CF, spc_pw_CF, fi, ft, dataDir)
    
    ##### Acor_calc_and_draw ######
    if draw_Acor_flg==1:
        Acor_env_tx = Xcor_func(t_array, txwv, txwv)
        Acor_env_tx_CF = Xcor_func(t_array, txwv_CF, txwv_CF)
        draw_wv("Acor", t_array, Acor_env_tx, min(t_array), max(t_array), min(Acor_env_tx)*1.1, max(Acor_env_tx)*1.1, fi, ft, dataDir)
        draw_wv("Acor_CF", t_array, Acor_env_tx_CF, min(t_array), max(t_array), min(Acor_env_tx_CF)*1.1, max(Acor_env_tx_CF)*1.1, fi, ft, dataDir)
        
    ##### tx_spectram(FFT)_draw ###############
    ###### transmit_wave("spectrum_tx")_and_reference_wave("spectrum_tx_CF")_draw######    
    if draw_tx_FFT_flg==1:
        txspc = np.abs(fft(txwv))
        txspc_CF = np.abs(fft(txwv_CF))
        f_array = np.linspace(0, fs, len(txwv))
        draw_spctrm("spectrum_tx", txspc, f_array/1000, 0, 2000, 0, fs/10000, fi, ft, dataDir)
        draw_spctrm("spectrum_tx_CF", txspc_CF, f_array/1000, 0, 2000, 0, fs/10000, fi, ft, dataDir)
  
    print("good")
    #return t_array, txwv, txwv_CF

def main():
    Fs,FFT_Smpl,Amplitude,Fi,Ft,Duration,Duration_CF,Duration_space,draw_tx_spc_flg,draw_Acor_flg,draw_tx_FFT_flg=Initialize()
    
    tx_design_view(Fs, FFT_Smpl, Amplitude, Fi, Ft, Duration, Duration_CF,Duration_space, True,True,True)

if __name__ == '__main__':
    main()