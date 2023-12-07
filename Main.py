from datetime import datetime
from pythonosc import dispatcher
from pythonosc import osc_server
import csv
import threading
import numpy as np
import pandas as pd
from collections import deque


ip = "192.168.99.214"
port = 5000

eeg_data = []
record_duration = 30
record_started = False
server = None 

def eeg_handler(address: str, *args):
    global eeg_data, record_started

    if not record_started:
        record_started = True
        threading.Timer(record_duration, stop_recording).start()

    eeg_data.append([datetime.now()] + list(args))
    
    if record_started:
        with open("TESTING.csv", "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Timestamp", "TP9", "AF7", "AF8", "TP10", "AUX"])
            csv_writer.writerows(eeg_data)

    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), *args)

def stop_recording():
    global eeg_data, record_started, server

    if record_started:
        # print("Data EEG telah disimpan dalam TESTING.csv")
        record_started = False
        if server is not None:
            server.shutdown()
            server.server_close()

if __name__ == "__main__":
    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/muse/eeg", eeg_handler)

    server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
    print("Listening on UDP port " + str(port))
    server.serve_forever()

#PreProcessing =====================================================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft

import spkit as sp
from spkit.data import load_data

data = pd.read_csv("TESTING.csv", delimiter=',')

data = data.copy()
timestamps = data['Timestamp']
data = data.drop(['Timestamp', 'AUX'], axis=1)

X = data.values

ch_names = ['TP9', 'AF7', 'AF8', 'TP10']
fs = 256

# FILTER BAND PASS
lowcut = 0.5  # Frekuensi cut-off bawah
highcut = 50  # Frekuensi cut-off atas

Xf = sp.filter_X(X, band=[lowcut, highcut], btype='bandpass', verbose=0)
Xf.shape
Xf.shape[0] / fs
t = np.arange(Xf.shape[0]) / fs
ch_names = ['TP9', 'AF7', 'AF8', 'TP10']

# FILTER ICA (Independet Component Analys)
AF_channels = ['AF7', 'AF8']
AF_ch_index = [ch_names.index(ch) for ch in AF_channels]
F_channels = []
F_ch_index = [ch_names.index(ch) for ch in F_channels]

sp.eeg.ICA_filtering
XR = sp.eeg.ICA_filtering(
    Xf.copy(),
    winsize=256,               
    ICA_method='fastica',  
    kur_thr=2,                 
    corr_thr=0.8,            
    AF_ch_index=AF_ch_index,   
    F_ch_index=F_ch_index,    
    verbose=True             
)
XR.shape
Xf.shape, XR.shape
Hasil = XR
data['Timestamp'] = timestamps

df_Hasil = pd.DataFrame({
    'Timestamp': timestamps, 
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
df_Hasil.to_csv("TESTINGICA.csv", index=False)  # Ganti nama file sesuai kebutuhan

# FILTER OUTLIER REMOVER (Independet Component Analys)
df = pd.read_csv("TESTINGICA.csv")
columns = ["TP9", "AF7", "AF8", "TP10"]
threshold = 3
outlier_indices = []
for column in columns:
    dataout = df[column]
    mean = np.mean(dataout)
    std = np.std(dataout)
    
    for i in range(len(dataout)):
        z = (dataout[i] - mean) / std
        if abs(z) > threshold:
            outlier_indices.append(i)
df_cleaned = df.drop(outlier_indices)

df['Timestamp'] = timestamps
df_cleaned.to_csv("TESTINGBERSIH.csv", index=False)

#Ekstraksi Fitur =====================================================================================================
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.integrate import simps
from scipy.signal import welch
import seaborn as sns
import numpy as np

data = pd.read_csv("TESTINGBERSIH.csv")

sampling_rate = 256
nperseg = 256 
noverlap = 128 
nfft = 1024  

tp9_data = data['TP9'].values
af7_data = data['AF7'].values
af8_data = data['AF8'].values
tp10_data = data['TP10'].values

def fft_feature_extraction(data, sampling_rate):
    N = len(data)
    normalize = N/2
    fft_result = fft(data)
    freq = fftfreq(N, 1 / sampling_rate)
    amplitude = np.abs(fft_result) / normalize
    
    return freq, amplitude

def extract_frequency_bands(freq, amplitude):
    delta_band = (0.5, 4.0)
    theta_band = (4.0, 8.0)
    alpha_band = (8.0, 13.0)
    beta_band = (13.0, 30.0)
    gamma_band = (30.0, 40.0)  

    delta_power = simps(amplitude[(freq >= delta_band[0]) & (freq <= delta_band[1])], dx=1/sampling_rate)
    theta_power = simps(amplitude[(freq >= theta_band[0]) & (freq <= theta_band[1])], dx=1/sampling_rate)
    alpha_power = simps(amplitude[(freq >= alpha_band[0]) & (freq <= alpha_band[1])], dx=1/sampling_rate)
    beta_power = simps(amplitude[(freq >= beta_band[0]) & (freq <= beta_band[1])], dx=1/sampling_rate)
    gamma_power = simps(amplitude[(freq >= gamma_band[0]) & (freq <= gamma_band[1])], dx=1/sampling_rate)

    return delta_power, theta_power, alpha_power, beta_power, gamma_power

freq_tp9, amplitude_tp9 = fft_feature_extraction(tp9_data, sampling_rate)
delta_tp9, theta_tp9, alpha_tp9, beta_tp9, gamma_tp9 = extract_frequency_bands(freq_tp9, amplitude_tp9)
freq_af7, amplitude_af7 = fft_feature_extraction(af7_data, sampling_rate)
delta_af7, theta_af7, alpha_af7, beta_af7, gamma_af7 = extract_frequency_bands(freq_af7, amplitude_af7)
freq_af8, amplitude_af8 = fft_feature_extraction(af8_data, sampling_rate)
delta_af8, theta_af8, alpha_af8, beta_af8, gamma_af8 = extract_frequency_bands(freq_af8, amplitude_af8)
freq_tp10, amplitude_tp10 = fft_feature_extraction(tp10_data, sampling_rate)
delta_tp10, theta_tp10, alpha_tp10, beta_tp10, gamma_tp10 = extract_frequency_bands(freq_tp10, amplitude_tp10)

eeg_columns = ['TP9', 'AF7', 'AF8', 'TP10']

delta_power = np.zeros(len(eeg_columns))
theta_power = np.zeros(len(eeg_columns))
alpha_power = np.zeros(len(eeg_columns))
beta_power = np.zeros(len(eeg_columns))
gamma_power = np.zeros(len(eeg_columns))

for i, col in enumerate(eeg_columns):
    eeg_data = data[col]
    frequencies, psd = welch(eeg_data, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

    delta_band = (0.5, 4)
    theta_band = (4, 8)
    alpha_band = (8, 13)
    beta_band = (13, 30)
    gamma_band = (30, 40)

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

# Gabungkan nilai Delta, Theta, Alpha, Beta, dan Gamma dari semua kanal
delta_total = np.sum(delta_power)
theta_total = np.sum(theta_power)
alpha_total = np.sum(alpha_power)
beta_total = np.sum(beta_power)
gamma_total = np.sum(gamma_power)

# Konversi ke dB (Desibel)
delta_power_db = 10 * np.log10(np.abs(delta_total))
theta_power_db = 10 * np.log10(np.abs(theta_total))
alpha_power_db = 10 * np.log10(np.abs(alpha_total))
beta_power_db = 10 * np.log10(np.abs(beta_total))
gamma_power_db = 10 * np.log10(np.abs(gamma_total))

# Data delta, theta, alpha, beta, gamma
delta_values = [delta_tp9, delta_tp10, delta_af7, delta_af8]
theta_values = [theta_tp9, theta_tp10, theta_af7, theta_af8]
alpha_values = [alpha_tp9, alpha_tp10, alpha_af7, alpha_af8]
beta_values = [beta_tp9, beta_tp10, beta_af7, beta_af8]
gamma_values = [gamma_tp9, gamma_tp10, gamma_af7, gamma_af8]

# Hitung mean dari delta, theta, alpha, beta, gamma power untuk masing-masing kanal
mean_delta_power = [np.mean(delta_values[i]) for i in range(len(delta_values))]
mean_theta_power = [np.mean(theta_values[i]) for i in range(len(theta_values))]
mean_alpha_power = [np.mean(alpha_values[i]) for i in range(len(alpha_values))]
mean_beta_power = [np.mean(beta_values[i]) for i in range(len(beta_values))]
mean_gamma_power = [np.mean(gamma_values[i]) for i in range(len(gamma_values))]

# Hitung rata-rata Delta, Theta, Alpha, Beta, dan Gamma power untuk semua kanal
all_mean_delta_power = np.mean(mean_delta_power)
all_mean_theta_power = np.mean(mean_theta_power)
all_mean_alpha_power = np.mean(mean_alpha_power)
all_mean_beta_power = np.mean(mean_beta_power)
all_mean_gamma_power = np.mean(mean_gamma_power)

# Menghitung standar deviasi dari Delta, Theta, Alpha, Beta, dan Gamma power untuk semua kanal
sd_delta_values = np.std(delta_values)
sd_theta_values = np.std(theta_values)
sd_alpha_values = np.std(alpha_values)
sd_beta_values = np.std(beta_values)
sd_gamma_values = np.std(gamma_values)

#STANDAR DEVIASI GENGSSSSS
delta_power_sd = np.std(delta_power)
theta_power_sd = np.std(theta_power)
alpha_power_sd = np.std(alpha_power)
beta_power_sd = np.std(beta_power)
gamma_power_sd = np.std(gamma_power)

data = {
    'Delta_Value': [all_mean_delta_power], 'Theta_Value': [all_mean_theta_power], 'Alpha_Value': [all_mean_alpha_power], 'Beta_Value': [all_mean_beta_power], 'Gamma_Value': [all_mean_gamma_power],
    # 'Delta_Power': [delta_power_db], 'Theta_Power': [theta_power_db], 'Alpha_Power': [alpha_power_db], 'Beta_Power': [beta_power_db], 'Gamma_Power': [gamma_power_db],
    'SD_Delta_Value': [sd_delta_values], 'SD_Theta_Value': [sd_theta_values], 'SD_Alpha_Value': [sd_alpha_values], 'SD_Beta_Value': [sd_beta_values], 'SD_Gamma_Value': [sd_gamma_values]
    # 'SD_Delta_Power': [delta_power_sd], 'SD_Theta_Power': [theta_power_sd], 'SD_Alpha_Power': [alpha_power_sd], 'SD_Beta_Power': [beta_power_sd], 'SD_Gamma_Power': [gamma_power_sd]
}

df = pd.DataFrame(data)
df.to_csv('TestingFix.csv', index=False, float_format='%.6f')

#Klasifikasi Testing =====================================================================================================
import pandas as pd
import joblib

data_to_classify = pd.read_csv("TestingFix.csv")

knn = joblib.load('knn_model.joblib')
predictions = knn.predict(data_to_classify)
print("Hasil prediksi:")
for prediction in predictions:
    print(prediction)