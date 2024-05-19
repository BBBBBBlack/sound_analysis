# 对数据集提取语谱图（spectrogram）并保存数据集和标签集为data.npy,label.npy
import os

import cv2
import librosa
import numpy as np
from matplotlib import pyplot as plt


def get_spectrogram(path):
    data, fs = librosa.load(path, sr=None, mono=True)
    spect = librosa.stft(data, n_fft=1024, hop_length=320, win_length=1024)
    # print(spect.shape)
    # 画语谱图
    plt.matshow(spect.astype(np.float), cmap='viridis')
    plt.ylabel('Frequency')
    plt.xlabel('Time(s)')
    plt.title('Spectrogram')
    plt.colorbar()
    plt.show()
    return spect


def extract_features():
    data_path = "data"
    labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    print("标签名：", labels)

    total_data = []
    total_label = []

    for label in labels:
        label_path = data_path + "\\" + label
        wav_names = os.listdir(label_path)
        for wav_name in wav_names:
            if wav_name.endswith(".wav"):
                wav_path = label_path + "\\" + wav_name
                # print(wav_path)
                spect = get_spectrogram(wav_path)
                spect = np.abs(spect)
                spect = cv2.resize(spect, (28, 28))
                total_data.append(spect)
                total_label.append(labels.index(label))

    total_data = np.array(total_data)
    total_label = np.array(total_label)
    print(total_data.shape)
    print(total_label.shape)
    np.save("data.npy", total_data)
    np.save("label.npy", total_label)


if __name__ == '__main__':
    extract_features()
