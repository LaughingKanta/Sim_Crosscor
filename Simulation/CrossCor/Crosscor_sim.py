# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 14:56:18 2021

@author: robat
Crosscorrelation clean
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.signal import hilbert
import scipy.signal as signal
import pathlib
import math

#実装用パラメータ設定
def initilize(): 
    Boxnum=32768
    amp=32767. / 2.#音圧
    fr=192000 #サンプリング周波数
    fpgafreq=1000000 #点数？
    dur=0.010  #duration
    Fstart=80000#開始周波数
    FEnd=40000#終端周波数
    BatCallConstant=0.005#パラメータ
    delay=0 
    
    sig2=[0]*Boxnum
    crosscor=[0]*Boxnum
    return Fstart,FEnd,BatCallConstant,amp,fr,fpgafreq,dur,Boxnum,delay,sig2,crosscor

#コウモリFM型信号作成
def makeFMsound(fStart,fEnd,BatCallConstant,amp,fs,dur):
    pi =np.pi
    nframes=int(dur*fs+1)
    arg= (BatCallConstant*fEnd)/fStart
    call=[]
    fStart=fStart/100
    fEnd=fEnd/100
    for i in range(nframes):
        t = float(i)/fs*100
        call.append(amp*np.sin(2.*pi*((fStart/(fStart-BatCallConstant*fEnd))*((fStart-fEnd)*np.float_power(arg, t)/math.log(arg)+(1-BatCallConstant)*fEnd*t))))
    return call

##相互相関する二つの信号を作成
def makesound(fStart,fEnd,BatCallConstant,amp,fs,dur,Boxnum,delay):
    call=makeFMsound(fStart, fEnd, BatCallConstant, amp, fs, dur)
    sig1=[]
    sig2=[]
    sig1=np.append(sig1,call)
    sig1=np.append(sig1,[0]*(Boxnum-(int(dur*fs+1))))
    sig2=np.append(sig2,[0]*delay)
    sig2=np.append(sig2,call)
    sig2=np.append(sig2,[0]*(Boxnum-delay-(int(dur*fs+1))))
    return sig1,sig2

## 相互相関(自己相関)山田さんfunction使いました。
def Xcor_func(sig1,sig2):
    sig1_ft=fft(sig1)
    crwv=ifft(fft(sig2)* np.conj(sig1_ft)).real
    hlbwv = hilbert(crwv)
    env = np.abs(hlbwv) 
    return env
#もともとの相互相関のfunction
"""
def spc_fumc(sig1,sig2):
    sig1_ft=fft(sig1)
    sig2_ft=fft(sig2)
    return sig1_ft,sig2_ft

def multiplySpecs(sig1,sig2,Boxnum):
    sig1_ft,sig2_ft=spc_func(sig1,sig2)
    mix_FT=[]
    hil_FT=[]
    for i in range(Boxnum):
        mix_FT=np.append(mix_FT,np.conjugate(sig1_FT[i]*sig2_FT[i]))
        if i<=Boxnum/2:
            hil_FT=np.append(hil_FT,complex(np.imag(mix_FT[i]),np.real(-mix_FT[i])))
        else:
            hil_FT=np.append(hil_FT,complex(np.imag(-mix_FT[i]),np.real(mix_FT[i])))
    return mix_FT,hil_FT

def calculateEnvelope(mix_FT,hil_FT,Boxnum,fpgafreq):
    CrossCor=[]
    mix_FT_inv=np.fft.ifft(mix_FT)
    hil_FT_inv=np.fft.ifft(hil_FT)
    for i in range(Boxnum):
        CrossCor=np.append(CrossCor,np.sqrt(np.real(mix_FT_inv[i])*np.real(mix_FT_inv[i])+np.real(hil_FT_inv[i])*np.real(hil_FT_inv[i])))
    #CrossCor=np.roll(CrossCor,-(Boxnum//2))
    time = np.arange(Boxnum) / float(fpgafreq)
    return CrossCor,time
"""
## Plot
def plot_cross_only(env):
    plt.plot(env)
    plt.title("Crosscor")
    plt.xlabel("data")
    plt.ylabel("Amplitude")
    plt.pause(0.1)
    plt.savefig("./output/plot_corr_10ms.png")

    


    
if __name__ == '__main__':
    fStart,fEnd,BatCallConstant,amp,fr,fpgafreq,dur,Boxnum,delay,sig2,crosscor=initilize()
    sig1,sig2=makesound(fStart,fEnd,BatCallConstant,amp,fpgafreq,dur,Boxnum,delay)
    env=Xcor_func(sig1, sig2)
    plot_cross_only(env)
    