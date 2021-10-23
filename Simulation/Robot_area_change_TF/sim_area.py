# -*- coding: utf-8 -*-
"""
Created on October 4 17:01:07 2021

@author: kanta
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.signal import hilbert
import scipy.signal as signal
import pathlib
import math
import random
def Initialize():
    #sensing_setting
    Fs=1000000 #[Hz]sampling_f
    FFT_Smpl=2**17 #Number_of_samplings_for_sensing[smpls]
    
    #pulse_design_setting
    Amplitude = 32767. / 2.   #pulse Amplitude: 
    Fi = 80000    #: initial frequency [Hz]
    Ft = 40000  #: terminal frequency [Hz]
    Duration =0.0075 #LPM_duration [s]
    Duration_CF =0.0025 # CF_duration [s]
    Duration_space = 0.01
    BatcallConstant=0.005#パラメーた
    draw_tx_spc_flg=True
    draw_Acor_flg=True
    draw_tx_FFT_flg=True   
    return Fs,FFT_Smpl,Amplitude,Fi,Ft,Duration,Duration_CF, Duration_space,BatcallConstant,draw_tx_spc_flg,draw_Acor_flg,draw_tx_FFT_flg
    
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

def add_pulse(fs, fi, ft, duration,duration_CF,duration_space,FFT_smpl,robotfreq,robotnum,ownfreq):
    pulsesum=[]
    Ts=1.0/fs #[Hz]sampling_period
    t_start = 0.0
    t_array = np.arange(t_start, FFT_smpl*Ts, Ts)
    n_dur_space=int(duration_space/Ts)+1
    pulse_space=t_array[:n_dur_space]
    pulsesum=np.append(pulsesum,np.zeros(len(pulse_space)))
    for i in range(robotnum):
        if robotfreq[i]<ownfreq:
            pulse_other=[]
            pulse_other=np.append(pulse_other,makeLPM_CFmodel(fs, fi, robotfreq[i], duration,duration_CF,FFT_smpl))
            pulse_other=np.append(pulse_other,np.zeros(len(pulse_space)))
        
            pulsesum=np.append(pulsesum,pulse_other)
       
    pulsesum=np.append(pulsesum,np.zeros(len(t_array)-len(pulsesum)))
    
    return pulsesum 

def makeWeighing(freq_array,ownfreq,max_freq,fft_pulsesum):
    a=0.1
    Wfft_pulsesum=[]
    for i in range((max_freq)):
        weight = math.exp((freq_array(i)-ownfreq)/(ownfreq*a))
        Wfft_pulsesum = np.append(Wfft_pulsesum,fft_pulsesum(i)*weight(i))
    
    return Wfft_pulsesum
def Area_calculation(fs, fi, ft, duration,duration_CF,duration_space,FFT_smpl):
    nsim=1000
    Area = []
    N_robot_low =[]
    for i in range(nsim):
        
            
        robotnum = random.randint(1,10) #ロボットの数をランダム
        #robotnum=4
        robotfreq=[]
        for i in range(robotnum):
            i= i*1000
            robotfreq=np.append(robotfreq, 40000+i)
            

        #robotfreq=[42000,44000,46000,48000]
    
        freq_array=np.linspace(0,fs,FFT_smpl)
        ownid =random.randint(0,robotnum-1)
        #ownid = (robotnum) #自分自身のIndex
        ownfreq = (robotfreq[ownid]) #自分自身のTF
        pulsesum=list()
        pulsesum = add_pulse(fs,fi,ft,duration,duration_CF,duration_space,FFT_smpl,robotfreq,robotnum,ownfreq)
        ##FFT
        fft_pulsesum=np.abs(fft(pulsesum))
        #面積する範囲を決める
        min_freq=round((ownfreq-5000)/(fs/FFT_smpl))+1
        max_freq=round(ownfreq+100/(fs/FFT_smpl))
       
        #重みづけ
       # Wfft_pulsesum=makeWeighing(freq_array, ownfreq, max_freq,fft_pulsesum)
        #print(len(Wfft_pulsesum))
        print(len(fft_pulsesum))
        s=0
        
        for i in range(min_freq,max_freq+1):
            s=s+fft_pulsesum[i]
            
       
        Area=np.append(Area,s)
        count=0
        for i in range(len(robotfreq)):
            if robotfreq[i]<ownfreq:
                count=count+i
                
        N_robot_low=np.append(N_robot_low,count)
            
    return N_robot_low,Area
            
#    Area = np.sum(fft_pulsesum(min_freq:max_freq))
def plot_area(fs, fi, ft, duration,duration_CF,duration_space,FFT_smpl):
    x,y=Area_calculation(fs, fi, ft, duration, duration_CF, duration_space, FFT_smpl)
    plt.plot(x,y,'o')
    plt.show()

    
  
    
    
    
    
    
    
    
    
    


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
    f=plt.figure(figsize=(8.0,8.0))
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
    Fs,FFT_Smpl,Amplitude,Fi,Ft,Duration,Duration_CF,Duration_space ,BatcallConstant,draw_tx_spc_flg,draw_Acor_flg,draw_tx_FFT_flg =Initialize()
    wvcf=makeLPM_CFmodel(Fs, Fi, Ft, Duration,Duration_CF,FFT_Smpl)
    Ts=1.0/Fs
    t_array = np.arange(0.0, FFT_Smpl*Ts, Ts)   
    dataDir = pathlib.Path('Output/')
    print(Fi, Ft)
    spc_f_cf, spc_t_cf, spc_pw_cf = signal.spectrogram(wvcf, Fs, nperseg=1024, noverlap=int(1024*0.98))
    draw_spectrogram("tx_cf", spc_t_cf, spc_f_cf, spc_pw_cf, Fi, Ft,dataDir)
  
    lpmspc = np.abs(fft(wvcf))
    f_array = np.linspace(0, Fs, len(wvcf))
    draw_spctrm("spectrum", lpmspc, f_array/1000, 0, 2000, 0, Fs/10000, Fi, Ft, dataDir)
    plot_area(Fs, Fi, Ft, Duration,Duration_CF,Duration_space,FFT_Smpl)


if __name__ == '__main__':
    main()
    
    