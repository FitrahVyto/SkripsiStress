import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.integrate import simps
from scipy.signal import welch
import seaborn as sns
import numpy as np

# Membaca file CSV
data = pd.read_csv('D:\KULIAH\SEMESTER 7 SKRIPSI\DATA SPLIT\P01_CANTIKA\P01_CANTIKA_HS.csv')

# Konfigurasi metode Welch
sampling_rate = 256  # Frekuensi sampel data EEG Anda
nperseg = 256  # Jumlah sampel dalam setiap segment
noverlap = 128  # Jumlah sampel yang tumpang tindih antara segment
nfft = 1024  # Jumlah sampel dalam FFT

# Pilih kolom yang berisi data sensor Muse (misalnya, TP9, AF7, AF8, TP10)
tp9_data = data['TP9'].values
af7_data = data['AF7'].values
af8_data = data['AF8'].values
tp10_data = data['TP10'].values

# Transformasi Fourier Cepat (FFT) untuk mengubah data waktu menjadi domain frekuensi
def fft_feature_extraction(data, sampling_rate):
    N = len(data)
    normalize = N/2
    fft_result = fft(data)
    freq = fftfreq(N, 1 / sampling_rate)
    amplitude = np.abs(fft_result) / normalize
    
    return freq, amplitude

def extract_frequency_bands(freq, amplitude):
    # Rentang frekuensi (dalam Hz)
    delta_band = (0.5, 4.0)
    theta_band = (4.0, 8.0)
    alpha_band = (8.0, 13.0)
    beta_band = (13.0, 30.0)
    gamma_band = (30.0, 40.0)  # Sesuaikan dengan kebutuhan Anda

    # Ekstraksi amplitudo dalam rentang frekuensi
    delta_power = simps(amplitude[(freq >= delta_band[0]) & (freq <= delta_band[1])], dx=1/sampling_rate)
    theta_power = simps(amplitude[(freq >= theta_band[0]) & (freq <= theta_band[1])], dx=1/sampling_rate)
    alpha_power = simps(amplitude[(freq >= alpha_band[0]) & (freq <= alpha_band[1])], dx=1/sampling_rate)
    beta_power = simps(amplitude[(freq >= beta_band[0]) & (freq <= beta_band[1])], dx=1/sampling_rate)
    gamma_power = simps(amplitude[(freq >= gamma_band[0]) & (freq <= gamma_band[1])], dx=1/sampling_rate)

    return delta_power, theta_power, alpha_power, beta_power, gamma_power

# Ekstraksi fitur dalam berbagai rentang frekuensi untuk setiap kanal
freq_tp9, amplitude_tp9 = fft_feature_extraction(tp9_data, sampling_rate)
delta_tp9, theta_tp9, alpha_tp9, beta_tp9, gamma_tp9 = extract_frequency_bands(freq_tp9, amplitude_tp9)
freq_af7, amplitude_af7 = fft_feature_extraction(af7_data, sampling_rate)
delta_af7, theta_af7, alpha_af7, beta_af7, gamma_af7 = extract_frequency_bands(freq_af7, amplitude_af7)
freq_af8, amplitude_af8 = fft_feature_extraction(af8_data, sampling_rate)
delta_af8, theta_af8, alpha_af8, beta_af8, gamma_af8 = extract_frequency_bands(freq_af8, amplitude_af8)
freq_tp10, amplitude_tp10 = fft_feature_extraction(tp10_data, sampling_rate)
delta_tp10, theta_tp10, alpha_tp10, beta_tp10, gamma_tp10 = extract_frequency_bands(freq_tp10, amplitude_tp10)

# OPSIONAL
# Menampilkan hasil ekstraksi fitur TP9
print("Delta Value (TP9):", delta_tp9)
print("Theta Value (TP9):", theta_tp9)
print("Alpha Value (TP9):", alpha_tp9)
print("Beta Value (TP9):", beta_tp9)
print("Gamma Value (TP9):", gamma_tp9)

# Menampilkan hasil ekstraksi fitur AF7
print("Delta Value (AF7):", delta_af7)
print("Theta Value (AF7):", theta_af7)
print("Alpha Value (AF7):", alpha_af7)
print("Beta Value (AF7):", beta_af7)
print("Gamma Value (AF7):", gamma_af7)

# Menampilkan hasil ekstraksi fitur AF8
print("Delta Value (AF8):", delta_af8)
print("Theta Value (AF8):", theta_af8)
print("Alpha Value (AF8):", alpha_af8)
print("Beta Value (AF8):", beta_af8)
print("Gamma Value (AF8):", gamma_af8)

# Menampilkan hasil ekstraksi fitur TP10
print("Delta Value (TP10):", delta_tp10)
print("Theta Value (TP10):", theta_tp10)
print("Alpha Value (TP10):", alpha_tp10)
print("Beta Value (TP10):", beta_tp10)
print("Gamma Value (TP10):", gamma_tp10)

# Fungsi untuk menghitung daya spektral
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

# OPSIONAL
for i, col in enumerate(eeg_columns):
    print(f'Kanal {col}:')
    print(f'Power of Delta: {delta_power[i]:.4f}')
    print(f'Power of Theta: {theta_power[i]:.4f}')
    print(f'Power of Alpha: {alpha_power[i]:.4f}')
    print(f'Power of Beta: {beta_power[i]:.4f}')
    print(f'Power of Gamma: {gamma_power[i]:.4f}')

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

print(f'Power of Delta (Total, Absolute dB): {delta_power_db:.4f}')
print(f'Power of Theta (Total, Absolute dB): {theta_power_db:.4f}')
print(f'Power of Alpha (Total, Absolute dB): {alpha_power_db:.4f}')
print(f'Power of Beta (Total, Absolute dB): {beta_power_db:.4f}')
print(f'Power of Gamma (Total, Absolute dB): {gamma_power_db:.4f}')

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

# Tampilkan nilai rata-rata untuk semua kanal
print("All Mean Delta Value:", all_mean_delta_power)
print("All Mean Theta Value:", all_mean_theta_power)
print("All Mean Alpha Value:", all_mean_alpha_power)
print("All Mean Beta Value:", all_mean_beta_power)
print("All Mean Gamma Value:", all_mean_gamma_power)

# Menghitung standar deviasi dari Delta, Theta, Alpha, Beta, dan Gamma power untuk semua kanal
sd_delta_values = np.std(delta_values)
sd_theta_values = np.std(theta_values)
sd_alpha_values = np.std(alpha_values)
sd_beta_values = np.std(beta_values)
sd_gamma_values = np.std(gamma_values)

# Tampilkan nilai standar deviasi
print("SD Delta Values:", sd_delta_values)
print("SD Theta Values:", sd_theta_values)
print("SD Alpha Values:", sd_alpha_values)
print("SD Beta Values:", sd_beta_values)
print("SD Gamma Values:", sd_gamma_values)

#STANDAR DEVIASI GENGSSSSS
# Menghitung SD dari Power Bands (Total, Absolute dB)
delta_power_sd = np.std(delta_power)
theta_power_sd = np.std(theta_power)
alpha_power_sd = np.std(alpha_power)
beta_power_sd = np.std(beta_power)
gamma_power_sd = np.std(gamma_power)

# Tampilkan SD Delta, Theta, Alpha, Beta, dan Gamma Power
print("SD Delta Power:", delta_power_sd)
print("SD Theta Power:", theta_power_sd)
print("SD Alpha Power:", alpha_power_sd)
print("SD Beta Power:", beta_power_sd)
print("SD Gamma Power:", gamma_power_sd)

# Membuat dataframe dengan nilai-nilai yang ingin disimpan
data = {
    'Delta_Value': [all_mean_delta_power], 'Theta_Value': [all_mean_theta_power], 'Alpha_Value': [all_mean_alpha_power], 'Beta_Value': [all_mean_beta_power], 'Gamma_Value': [all_mean_gamma_power],
    'Delta_Power': [delta_power_db], 'Theta_Power': [theta_power_db], 'Alpha_Power': [alpha_power_db], 'Beta_Power': [beta_power_db], 'Gamma_Power': [gamma_power_db],
    'SD_Delta_Value': [sd_delta_values], 'SD_Theta_Value': [sd_theta_values], 'SD_Alpha_Value': [sd_alpha_values], 'SD_Beta_Value': [sd_beta_values], 'SD_Gamma_Value': [sd_gamma_values],
    'SD_Delta_Power': [delta_power_sd], 'SD_Theta_Power': [theta_power_sd], 'SD_Alpha_Power': [alpha_power_sd], 'SD_Beta_Power': [beta_power_sd], 'SD_Gamma_Power': [gamma_power_sd],
    'TP9_Delta': [delta_tp9], 'TP9_Theta': [theta_tp9], 'TP9_Alpha': [alpha_tp9], 'TP9_Beta': [beta_tp9], 'TP9_Gamma': [gamma_tp9],
    'AF7_Delta': [delta_af7], 'AF7_Theta': [theta_af7], 'AF7_Alpha': [alpha_af7], 'AF7_Beta': [beta_af7], 'AF7_Gamma': [gamma_af7],
    'AF8_Delta': [delta_af8], 'AF8_Theta': [theta_af8], 'AF8_Alpha': [alpha_af8], 'AF8_Beta': [beta_af8], 'AF8_Gamma': [gamma_af8],
    'TP10_Delta': [delta_tp10], 'TP10_Theta': [theta_tp10], 'TP10_Alpha': [alpha_tp10], 'TP10_Beta': [beta_tp10], 'TP10_Gamma': [gamma_tp10],
    'TP9_Delta_Power': [delta_power[0]], 'TP9_Theta_Power': [theta_power[0]], 'TP9_Alpha_Power': [alpha_power[0]], 'TP9_Beta_Power': [beta_power[0]], 'TP9_Gamma_Power': [gamma_power[0]],
    'AF7_Delta_Power': [delta_power[1]], 'AF7_Theta_Power': [theta_power[1]], 'AF7_Alpha_Power': [alpha_power[1]], 'AF7_Beta_Power': [beta_power[1]], 'AF7_Gamma_Power': [gamma_power[1]],
    'AF8_Delta_Power': [delta_power[2]], 'AF8_Theta_Power': [theta_power[2]], 'AF8_Alpha_Power': [alpha_power[2]], 'AF8_Beta_Power': [beta_power[2]], 'AF8_Gamma_Power': [gamma_power[2]],
    'TP10_Delta_Power': [delta_power[3]], 'TP10_Theta_Power': [theta_power[3]], 'TP10_Alpha_Power': [alpha_power[3]], 'TP10_Beta_Power': [beta_power[3]], 'TP10_Gamma_Power': [gamma_power[3]]
}

df = pd.DataFrame(data)

# Menyimpan dataframe ke dalam file CSV
df.to_csv('EX.csv', index=False, float_format='%.6f')