import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import pandas as pd

# Baca data EEG dari file CSV
data = pd.read_csv('BB.csv')  # Pastikan file 'data_eeg.csv' sesuai dengan data EEG Anda

# Pilih kolom EEG yang sesuai (TP9, AF7, AF8, TP10)
eeg_columns = ['TP9', 'AF7', 'AF8', 'TP10']

# Konfigurasi metode Welch
fs = 256  # Frekuensi sampel data EEG Anda
nperseg = 256  # Jumlah sampel dalam setiap segment
noverlap = 128  # Jumlah sampel yang tumpang tindih antara segment
nfft = 1024  # Jumlah sampel dalam FFT

# Inisialisasi variabel untuk menyimpan daya di masing-masing band
delta_power = np.zeros(len(eeg_columns))
theta_power = np.zeros(len(eeg_columns))
alpha_power = np.zeros(len(eeg_columns))
beta_power = np.zeros(len(eeg_columns))
gamma_power = np.zeros(len(eeg_columns))

# Loop untuk menghitung daya di masing-masing band untuk setiap kanal
for i, col in enumerate(eeg_columns):
    eeg_data = data[col]
    
    # Hitung PSD dengan metode Welch
    frequencies, psd = welch(eeg_data, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

    # Definisikan rentang frekuensi yang sesuai dengan band yang diinginkan
    delta_band = (0.5, 4)  # Delta (0.5 - 4 Hz)
    theta_band = (4, 8)    # Theta (4 - 8 Hz)
    alpha_band = (8, 13)   # Alpha (8 - 13 Hz)
    beta_band = (13, 30)   # Beta (13 - 30 Hz)
    gamma_band = (30, 100)  # Gamma (30 - 100 Hz)

    # Loop untuk menghitung daya di masing-masing band
    for j in range(len(frequencies)):
        if delta_band[0] <= frequencies[j] <= delta_band[1]:
            delta_power[i] += psd[j]
        elif theta_band[0] <= frequencies[j] <= theta_band[1]:
            theta_power[i] += psd[j]
        elif alpha_band[0] <= frequencies[j] <= alpha_band[1]:
            alpha_power[i] += psd[j]
        elif beta_band[0] <= frequencies[j] <= beta_band[1]:
            beta_power[i] += psd[j]
        elif gamma_band[0] <= frequencies[j] <= gamma_band[1]:
            gamma_power[i] += psd[j]

# Cetak data numerik
for i, col in enumerate(eeg_columns):
    print(f'Kanal {col}:')
    print(f'Power of Delta: {delta_power[i]:.4f}')
    print(f'Power of Theta: {theta_power[i]:.4f}')
    print(f'Power of Alpha: {alpha_power[i]:.4f}')
    print(f'Power of Beta: {beta_power[i]:.4f}')
    print(f'Power of Gamma: {gamma_power[i]:.4f}')
    
