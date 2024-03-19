import warnings

warnings.filterwarnings('ignore')
from htk.HTKFeat import MFCC_HTK
import os
import numpy as np
import matplotlib.pyplot as plt

data_path1 = os.path.join(os.getcwd(), '02_data\\man.wav')
data_path2 = os.path.join(os.getcwd(), '02_data\\woman.wav')
'''预处理'''
mfcc = MFCC_HTK()
signal = mfcc.load_raw_signal(data_path1)
sig_len = signal.size / 16000  # in seconds
plt.figure(figsize=(15, 4))
t = np.linspace(0, sig_len, signal.size)
plt.title('Woman\'s Signal')
plt.plot(t, signal)
# 频谱图
plt.figure(figsize=(15, 4))
plt.specgram(signal, Fs=16000, scale_by_freq=True, sides='default',
             scale='dB')
plt.title('Woman\'s Spectrogram')
plt.xlim(0, sig_len)

'''移除数据中心值'''
print("Before: " + str(np.mean(signal)))
signal = signal - np.mean(signal)
print("After: " + str(np.mean(signal)))
'''分帧'''
win_shift = 160
win_len = 400
sig_len = len(signal)
win_num = np.floor((sig_len - win_len) / win_shift).astype('int') + 1
wins = []
for w in range(win_num):
    s = w * win_shift
    e = s + win_len
    win = signal[s:e].copy()
    wins.append(win)
wins = np.asarray(wins)
'''预加重'''
'''增加某些频率范围内的信号强度，以改善信号与噪音之间的比率，提高信号的质量和清晰度'''
k = 0.97
h = [1, -k]
f = np.linspace(0, 8000, 257)
plt.figure()
plt.plot(f, np.abs(np.fft.rfft(h, n=512)))
plt.xlabel('Frequency')
plt.ylabel('Amplitude correction')
for win in wins:
    win -= np.hstack((win[0], win[:-1])) * k
plt.figure()
plt.plot(np.abs(np.fft.rfft(h, n=(wins.shape[0] - 1) * 2)), wins)
plt.xlabel('Amplitude correction')
plt.ylabel('Frequency')
plt.title('Woman\'s Pre-emphasis')
'''加窗'''
'''通过对信号进行窗函数的乘积来限制信号在时间域上的有效范围，以减少信号在频谱分析过程中的波动和混叠等问题'''
plt.figure(figsize=(12, 4))
plt.subplot(2, 1, 1)
# 矩形窗
rect = np.ones(400)
plt.stem(rect)
plt.xlim(-100, 500)
plt.ylim(-0.1, 1.1)
plt.title('Square window')
# 汉明窗
hamm = np.hamming(400)
plt.subplot(2, 1, 2)
plt.stem(hamm)
plt.xlim(-100, 500)
plt.ylim(-0.1, 1.1)
plt.title('Hamming function')
# 加窗后的频谱
plt.figure(figsize=(12, 3))
# plt.subplot(2, 1, 1)
for win in wins:
    win *= hamm
plt.plot(np.abs(np.log(np.abs(np.fft.rfft(hamm, n=(wins.shape[0] - 1) * 2)))), wins)
plt.xlim(-2, 35)
plt.xlabel('Frequency')
plt.ylabel('Amplitude (log)')
plt.show()
