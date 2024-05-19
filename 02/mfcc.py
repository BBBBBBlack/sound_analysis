import os

import matplotlib.pyplot as plt
import numpy as np
# from pre_process import pre_process

from htk.HTKFeat import MFCC_HTK

# 配置数据路径
data_path1 = os.path.join(os.getcwd(), 'D:\\xxx\\sound_analysis\\03\\train_dataset\\bird\\'
                                       'bird_4-2_1595274333851.wav')
data_path2 = os.path.join(os.getcwd(), '02_data\\woman.wav')
win_shift = 160
win_len = 400
mfcc = MFCC_HTK()
signal = mfcc.load_raw_signal(data_path1)
sig_len = len(signal)
win_num = np.floor((sig_len - win_len) / win_shift).astype('int') + 1
wins = []
for w in range(win_num):
    # t 每个窗的开始和结束
    s = w * win_shift
    e = s + win_len
    win = signal[s:e].copy()
    wins.append(win)
wins = np.asarray(wins)

'''快速傅里叶变换'''
fft_len = (2 ** (np.floor(np.log2(win_len)) + 1)).astype('int')
ffts = []
for win in wins:
    win = np.abs(np.fft.rfft(win, n=fft_len)[:-1])
    ffts.append(win)
ffts = np.asarray(ffts)
# 频谱图
plt.figure(figsize=(10, 5))
plt.pcolormesh(ffts.T, cmap='gray')
plt.xlim(0, win_num)
plt.ylim(0, fft_len / 2)

'''梅尔频谱倒频系数'''
freq2mel = lambda freq: 1127 * (np.log(1 + ((freq) / 700.0)))
f = np.linspace(0, 8000, 1000)
m = freq2mel(f)
plt.figure()
plt.plot(f, m)
plt.xlabel('Frequency')
plt.ylabel('Mel')
'''滤波器'''
# 三角滤波器
mfcc.create_filter(26)
plt.figure(figsize=(15, 3))
for f in mfcc.filter_mat.T:
    plt.plot(f)
plt.xlim(0, 256)
# 快速傅里叶变换的结果与滤波器乘积
melspec = []
for f in ffts:
    m = np.dot(f, mfcc.filter_mat)
    melspec.append(m)
melspec = np.asarray(melspec)
plt.figure(figsize=(15, 5))
plt.pcolormesh(melspec.T, cmap='gray')
plt.xlim(0, win_num)
plt.ylim(0, 26)
#  放大数据中的细节同时衰减高光
mels = np.log(melspec)
plt.figure(figsize=(15, 5))
plt.pcolormesh(mels.T, cmap='gray')
plt.xlim(0, win_num)
plt.ylim(0, 26)
''' 离散余弦变换'''
filter_num = 26
mfcc_num = 12
dct_base = np.zeros((filter_num, mfcc_num))
for m in range(mfcc_num):
    dct_base[:, m] = np.cos((m + 1) * np.pi / filter_num * (np.arange(filter_num) + 0.5))

plt.figure(figsize=(6, 3))
plt.pcolormesh(dct_base.T, cmap='gray')
plt.xlim(0, 24)
plt.ylim(0, 12)
plt.xlabel('Filters')
plt.ylabel('MFCCs')
# 将梅尔频谱图与 DCT 矩阵进行矩阵乘法以获得 MFCC
filter_num = 26
mfcc_num = 12
mfccs = []
for m in mels:
    c = np.dot(m, dct_base)
    mfccs.append(c)
mfccs = np.asarray(mfccs)
plt.figure(figsize=(15, 5))
plt.pcolormesh(mfccs.T, cmap='RdBu')  # cmap:RdBu,Oranges,YlGn,OrRd,gray，...
plt.xlim(0, win_num)
plt.ylim(0, mfcc_num)
# 归一化
mfnorm = np.sqrt(2.0 / filter_num)
mfccs *= mfnorm
'''对比倒频谱能量与原始能量'''
raw_energy = []
for win in wins:
    raw_energy.append(np.log(np.sum(win ** 2)))
raw_energy = np.asarray(raw_energy)
ceps_energy = []
for m in mels:
    ceps_energy.append(np.sum(m) * mfnorm)
ceps_energy = np.asarray(ceps_energy)

plt.figure(figsize=(15, 5))
plt.plot(raw_energy, 'r', label='Raw energy')
plt.plot(ceps_energy, 'b', label='Cepstral energy')  # 倒频谱能量
plt.xlim(0, win_num)
plt.legend()
plt.show()
