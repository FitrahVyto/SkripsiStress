import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft

import spkit as sp
from spkit.data import load_data

data = pd.read_csv('/home/pifitrah/SkripsiVyto/SkripsiStress/DataTesting/Test_Fix_60_Attar.csv', delimiter=',')  # Ganti 'data_eeg.csv' dengan nama file Anda

data = data.copy()  # Buat salinan DataFrame
timestamps = data['Timestamp']
data = data.drop(['Timestamp', 'AUX'], axis=1)  # Hapus kolom waktu dari salinan DataFrame

X = data.values

ch_names = ['TP9', 'AF7', 'AF8', 'TP10']
fs = 256

# FILTER BAND PASS
lowcut = 0.5  # Frekuensi cut-off bawah
highcut = 50  # Frekuensi cut-off atas

# Buat bandpass filter
Xf = sp.filter_X(X, band=[lowcut, highcut], btype='bandpass', verbose=0)

Xf.shape

Xf.shape[0] / fs
t = np.arange(Xf.shape[0]) / fs

# Definisikan nama kanal sesuai dengan data EEG Anda
ch_names = ['TP9', 'AF7', 'AF8', 'TP10']

# FILTER ICA (Independet Component Analys)

# Daftar kanal prefrontal (AF - First Layer of electrodes towards frontal lobe)
AF_channels = ['AF7', 'AF8']
AF_ch_index = [ch_names.index(ch) for ch in AF_channels]

# Daftar kanal frontal (F - second layer of electrodes)
F_channels = []
F_ch_index = [ch_names.index(ch) for ch in F_channels]

sp.eeg.ICA_filtering

# Terapkan Independent Component Analysis (ICA) untuk filtering
XR = sp.eeg.ICA_filtering(
    Xf.copy(),
    winsize=256,               # Ukuran jendela ICA
    ICA_method='fastica',  # Metode ICA
    kur_thr=2,                 # Ambang kurtosis
    corr_thr=0.8,              # Ambang korelasi
    AF_ch_index=AF_ch_index,   # Indeks kanal prefrontal
    F_ch_index=F_ch_index,     # Indeks kanal frontal
    verbose=True               # Tampilkan informasi verbose
)
XR.shape

Xf.shape, XR.shape

Hasil = XR

data['Timestamp'] = timestamps

# Buat DataFrame baru dengan data yang sudah di-restorasi
df_Hasil = pd.DataFrame({
    'Timestamp': timestamps,  # Tambahkan kembali kolom Timestamp
    # 'TP9_pre': eeg['TP9'],
    'TP9': Hasil[:, 0],
    # 'AF7_pre': eeg['AF7'],
    'AF7': Hasil[:, 1],
    # 'AF8_pre': eeg['AF8'],
    'AF8': Hasil[:, 2],
    # 'TP10_pre': eeg['TP10'],
    'TP10': Hasil[:, 3],
    # 'ICA_Comp_4': comps[:, 3]  # Ganti nama sesuai kebutuhan
})

# Simpan DataFrame ke file CSV
df_Hasil.to_csv('Fix_60_Attar_ICA.csv', index=False)  # Ganti nama file sesuai kebutuhan

# FILTER OUTLIER REMOVER (Independet Component Analys)

df = pd.read_csv("Fix_60_Attar_ICA.csv")

# Kolom yang akan dianalisis (meliputi tp9, af7, af8, tp10)
columns = ["TP9", "AF7", "AF8", "TP10"]

# Menentukan threshold Z-score menggunakan mean dan std
threshold = 3

# Mencari dan menghapus outlier
outlier_indices = []

for column in columns:
    dataout = df[column]
    mean = np.mean(dataout)
    std = np.std(dataout)
    
    for i in range(len(dataout)):
        z = (dataout[i] - mean) / std
        if abs(z) > threshold:
            outlier_indices.append(i)

# Menghapus baris yang mengandung outlier
df_cleaned = df.drop(outlier_indices)

df['Timestamp'] = timestamps

# Menyimpan data yang telah dibersihkan ke file CSV baru
df_cleaned.to_csv("Fix_60_Attar_PREPROCESSING.csv", index=False)
                
