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
    FFT_Smpl=2**15 #Number_of_samplings_for_sensing[smpls]
    
    #pulse_design_setting
    Amplitude = 32767. / 2.   #pulse Amplitude: 
    Fi = 80000    #: initial frequency [Hz]
    Ft = 40000  #: terminal frequency [Hz]
    Duration =0.01 #LPM_duration [s]
    Duration_CF =0.01 # CF_duration [s]
    Duration_mimic=0.01#mimicFM_duration[s]
    BatcallConstant=0.005#パラメータ
    draw_tx_spc_flg=True
    draw_Acor_flg=True
    draw_tx_FFT_flg=True   
    return Fs,FFT_Smpl,Amplitude,Fi,Ft,Duration,Duration_CF,Duration_mimic,BatcallConstant,draw_tx_spc_flg,draw_Acor_flg,draw_tx_FFT_flg
    
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

def makeCFmodel(fs, ft, duration_CF,FFT_smpl):
   Ts=1.0/fs #[Hz]sampling_period
   t_start = 0.0
   t_array = np.arange(t_start, FFT_smpl*Ts, Ts)
 
   n_dur=int(duration_CF/Ts)+1
   t= t_array[:n_dur] 
   Ts=1.0/fs #[Hz]sampling_period
   f=ft*t/t
   del_phase = 2*np.pi*f*Ts
   phase=0
   wave=np.zeros(len(del_phase))
    
   for i in range(len(del_phase)):
       phase = phase +del_phase[i]
       wave[i] = np.sin(phase)
    
   wave=np.append(wave, np.zeros(len(t_array)-len(wave)))
   return wave


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

def main():
    Fs,FFT_Smpl,Amplitude,Fi,Ft,Duration,Duration_CF,Duration_mimic,BatcallConstant,draw_tx_spc_flg,draw_Acor_flg,draw_tx_FFT_flg=Initialize()
    wvcf=makeCFmodel(Fs, Ft, Duration_CF,FFT_Smpl)
    wvlpm=makeLPMmodel(Fs, Fi, Ft, Duration,FFT_Smpl)
    Ts=1.0/Fs
    
   
    t_array = np.arange(0.0, FFT_Smpl*Ts, Ts)
    wave=np.zeros(len(t_array))
    for i in range(len(t_array)):
        wave[i]=wvcf[i]+wvlpm[i]
    


    

       
    dataDir = pathlib.Path('Output/below/')
    print(Fi, Ft)
    spc_f, spc_t, spc_pw = signal.spectrogram(wvcf, Fs, nperseg=1024, noverlap=int(1024*0.98))
    draw_spectrogram("tx_below", spc_t, spc_f, spc_pw, Fi, Ft, dataDir)   
    
    

if __name__ == '__main__':
    main()