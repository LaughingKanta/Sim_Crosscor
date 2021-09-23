# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:18:53 2021

@author: robat
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.signal import hilbert
import scipy.signal as signal
import pathlib
import math

def Initialize():
    #sensing_setting
    Fs=1000000 #[Hz]sampling_f
    FFT_Smpl=2**17 #Number_of_samplings_for_sensing[smpls]
    
    #pulse_design_setting
    Amplitude = 32767. / 2.   #pulse Amplitude: 
    Fi = 80000    #: initial frequency [Hz]
    Ft = 40000  #: terminal frequency [Hz]
    RobotNum=6
    Robotfreq=[40000,42000,44000,46000,48000,50000]
    Duration =0.0075 #LPM_duration [s]
    Duration_CF =0.0025 # CF_duration [s]
    Duration_mimic=0.01#mimicFM_duration[s]
    Duration_space=0.01
    BatcallConstant=0.005#パラメーた
    draw_tx_spc_flg=True
    draw_Acor_flg=True
    draw_tx_FFT_flg=True   
    return Fs,FFT_Smpl,Amplitude,Fi,Ft,RobotNum,Robotfreq,Duration,Duration_CF,Duration_mimic,Duration_space,BatcallConstant,draw_tx_spc_flg,draw_Acor_flg,draw_tx_FFT_flg
    
def makeLPMmodel(fs, fi, ft, duration,FFT_smpl):
    Ts=1.0/fs #[Hz]sampling_period
    t_start = 0.0
    t_array = np.arange(t_start, FFT_smpl*Ts, Ts)
    n_dur=int(duration/Ts)+1
    t= t_array[:n_dur] 
    k=(1-ft/fi)/(ft*duration)
    f=fi/(1+k*fi*t)
    del_phase = 2*np.pi*f*Ts
    phase=0
    wave=np.zeros(len(del_phase))
    
    for i in range(len(del_phase)):
        phase = phase +del_phase[i]
        wave[i] = np.sin(phase)
    
    wave=np.append(wave, np.zeros(len(t_array)-len(wave)))
    return wave


def makeLPM_CFmodel(fs, fi, ft, duration,duration_CF,FFT_smpl):

    Ts=1.0/fs #[Hz]sampling_period
    t_start = 0.0
    t_array = np.arange(t_start, FFT_smpl*Ts, Ts)
    n_dur=int(duration/Ts)+1
    n_dur_CF=int(duration_CF/Ts)+1
    t= t_array[:n_dur] 
    t_CF=t_array[n_dur:n_dur+n_dur_CF]
    k=(1-ft/fi)/(ft*duration)
    f=fi/(1+k*fi*t)
    del_phase = 2*np.pi*f*Ts
    f=ft*t_CF/t_CF
    del_phase_CF = 2*np.pi*f*Ts
    phase_CF=0
    phase=0
    wvlpm_cf=[]
    #wvlpm=np.zeros(len(del_phase))
    #wvcf=np.zeros(len(del_phase_CF))
    for i in range(len(del_phase)):
        phase = phase +del_phase[i]
        wvlpm_cf =np.append(wvlpm_cf,np.sin(phase))
    for i in range(len(del_phase_CF)):
        phase_CF=phase_CF+del_phase_CF[i]
        wvlpm_cf=np.append(wvlpm_cf,np.sin(phase_CF))
    #wvlpm_cf=np.append(wvlpm_cf, np.zeros(len(t_array)-len(wvlpm_cf)))
    return wvlpm_cf    

def add_pulse(fs, fi, ft, duration,duration_CF,duration_space,FFT_smpl,robotfreq,robotnum):
    pulsesum=[]
    Ts=1.0/fs #[Hz]sampling_period
    t_start = 0.0
    t_array = np.arange(t_start, FFT_smpl*Ts, Ts)
    n_dur_space=int(duration_space/Ts)+1
    pulse_space=t_array[:n_dur_space]
    pulsesum=np.append(pulsesum,np.zeros(len(pulse_space)))
    for i in range(robotnum):
        pulse_other=[]
        pulse_other=np.append(pulse_other,makeLPM_CFmodel(fs, fi, robotfreq[i], duration,duration_CF,FFT_smpl))
        pulse_other=np.append(pulse_other,np.zeros(len(pulse_space)))
    
        pulsesum=np.append(pulsesum,pulse_other)
       
    pulsesum=np.append(pulsesum,np.zeros(len(t_array)-len(pulsesum)))
    
    return pulsesum 

def Xcor_lpm_jamming__func(fs, fi, ft, duration,duration_CF,duration_space,FFT_smpl,robotfreq,robotnum):
    pulsesum=[]
    dataDir = pathlib.Path('Output/')
    wvlpm=makeLPMmodel(fs, fi, ft, duration,FFT_smpl)
    Ts=1.0/fs #[Hz]sampling_period
    t_start = 0.0
    t_array = np.arange(t_start, FFT_smpl*Ts, Ts)
    n_dur_space=int(duration_space/Ts)+1
    pulse_space=t_array[:n_dur_space]
    
    Xcor_other_array=np.zeros((robotnum, FFT_smpl))
    for i in range(robotnum):
        pulse_other=[]
        pulse_other=np.append(pulse_other,np.zeros(len(pulse_space)))
        pulse_other=np.append(pulse_other,makeLPM_CFmodel(fs, fi, robotfreq[i], duration,duration_CF,FFT_smpl))
        pulse_other=np.append(pulse_other,np.zeros(len(t_array)-len(pulse_other)))
        pulse_other=np.roll(pulse_other,len(pulse_space)*2*i)
        Xcor_other=Xcor_func(t_array,wvlpm,pulse_other)
        Xcor_other_array[i]=Xcor_other
       
   
    draw_wv_for_robot("Xcor_for",t_array*1000, Xcor_other_array, 0, 120, min(Xcor_other)*1.1, 4000, fi, ft, dataDir)
 
        
def draw_wv_for_robot(fig_name, x_data, y_data, x_min, x_max, y_min, y_max, f_init, f_termin, val):
    fig, (ax1) = plt.subplots(1,1, sharex=True, constrained_layout=True)
    for y in y_data:
        ax1.plot(x_data, y)
    plt.figure(figsize=(8.0, 3))
    ax1.set_xlim(0, x_max)
    ax1.set_ylim(y_min, y_max)
    #ax1.set_title(r"$f_i$ = {} kHz   $f_t$ = {} kHz".format(str(int(f_init/1000)), str(int(f_termin/1000))))
    ax1.set_ylabel("waveform")
   # ax1.set_xlabel("Time [s]")
    #plt.xticks(color="None")
    fig.savefig(val/'{}_{}_{}.png'.format(fig_name, str(int(f_init/1000)), str(int(f_termin/1000))))
    plt.show()
    fig.clear()
    

    ax1.clear()
    plt.close()    
    

def Xcor_func(t, Txwv, Rxwv):
    txspc = fft(Txwv)    
    crwv = ifft(fft(Rxwv) * np.conj(txspc)).real
    hlbwv = hilbert(crwv)
    env = np.abs(hlbwv) 
    return env

def draw_wv(fig_name, x_data, y_data, x_min, x_max, y_min, y_max, f_init, f_termin, val):
    fig, (ax1) = plt.subplots(1,1, sharex=True, constrained_layout=True)
    ax1.plot(x_data, y_data)
    x_data=x_data*1000
    ax1.set_xlim(0, x_max)
    ax1.set_ylim(y_min, y_max)
    plt.figure(figsize=(8.0, 3))
    #plt.xticks(color="None")
    ax1.set_title(r"$f_i$ = {} kHz  $f_t$ = {} kHz".format(str(int(f_init/1000)), str(int(f_termin/1000))))
    ax1.set_ylabel("waveform")
    ax1.set_xlabel("Time [ms]")
    fig.savefig(val/'{}_{}_{}.png'.format(fig_name, str(int(f_init/1000)), str(int(f_termin/1000))))
    plt.show()
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

def draw_spectrogram(fig_name, x_data, y_data, z_color, f_init, f_termin,val):
    plt.rcParams["font.size"] = 20
    x_data=x_data*1000
    y_data=y_data/1000
    y_max=100
    f=plt.figure(figsize=(8.0,2.0))
    plt.pcolormesh(x_data, y_data, z_color, cmap="hot")
    plt.title(r"$ f_i$ = {} kHz  $f_t$ = {} kHz".format(str(int(f_init/1000)), str(int(f_termin/1000))))
    plt.xlim([0,max(x_data)])
    plt.ylim([0,y_max])
    plt.xlabel("Time [ms]")
    #plt.xticks(color="None")
    plt.ylabel("Frequency [kHz]") 
    plt.colorbar()
    plt.savefig(val/'{}_spctrgm{}_{}.png'.format(fig_name, str(int(f_init/1000)), str(int(f_termin/1000))))
    plt.show()
    f.clear()
    plt.close()

def main():
    Fs,FFT_Smpl,Amplitude,Fi,Ft,RobotNum,Robotfreq,Duration,Duration_CF,Duration_mimic,Duration_space,BatcallConstant,draw_tx_spc_flg,draw_Acor_flg,draw_tx_FFT_flg =Initialize()
    wvcf=makeLPM_CFmodel(Fs, Fi, Ft, Duration,Duration_CF,FFT_Smpl)
    wvlpm=makeLPMmodel(Fs, Fi, Ft, Duration,FFT_Smpl)
    Ts=1.0/Fs
    t_array = np.arange(0.0, FFT_Smpl*Ts, Ts)   
    dataDir = pathlib.Path('Output/')
    pulsesum=add_pulse(Fs, Fi, Ft, Duration, Duration_CF, Duration_space, FFT_Smpl, Robotfreq, RobotNum)
    print(Fi, Ft)
    Xcor_lpm_jamming__func(Fs, Fi, Ft, Duration,Duration_CF,Duration_space,FFT_Smpl,Robotfreq,RobotNum)
    print("akan")
    spc_f_cf, spc_t_cf, spc_pw_cf = signal.spectrogram(wvcf, Fs, nperseg=1024, noverlap=int(1024*0.98))
    spc_f_lpm, spc_t_lpm, spc_pw_lpm = signal.spectrogram(wvlpm, Fs, nperseg=1024, noverlap=int(1024*0.98))
    spc_f, spc_t, spc_pw = signal.spectrogram(pulsesum, Fs, nperseg=1024, noverlap=int(1024*0.98))
    draw_spectrogram("tx", spc_t_lpm, spc_f_lpm, spc_pw_lpm, Fi, Ft,dataDir)
    draw_spectrogram("tx_cf", spc_t_cf, spc_f_cf, spc_pw_cf, Fi, Ft,dataDir)
    draw_spectrogram("tx_sum", spc_t, spc_f, spc_pw, Fi, Ft,dataDir)
    
    Xcor_env_tx_CF = Xcor_func(t_array, wvlpm, pulsesum)
    draw_wv("Xcor", t_array, Xcor_env_tx_CF, min(t_array), max(t_array), min(Xcor_env_tx_CF)*1.1, max(Xcor_env_tx_CF)*1.1, Fi, Ft, dataDir)
    
    sumspc = np.abs(fft(pulsesum))
    f_array = np.linspace(0, Fs, len(pulsesum))
    draw_spctrm("spectrum_sum", sumspc, f_array/1000, 0, 2500, 0, Fs/10000, Fi, Ft, dataDir)
    
    lpmspc = np.abs(fft(wvlpm))
    f_array = np.linspace(0, Fs, len(wvlpm))
    draw_spctrm("spectrum", lpmspc, f_array/1000, 0, 2000, 0, Fs/10000, Fi, Ft, dataDir)


if __name__ == '__main__':
    main()