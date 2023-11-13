import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.integrate import simps
from scipy.signal import welch
import seaborn as sns
import numpy as np  # Import NumPy

# Membaca file CSV
data = pd.read_csv('hasil_ica.csv')

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
    n = len(data)
    fft_result = fft(data)
    freq = fftfreq(n, 1 / sampling_rate)
    amplitude = abs(fft_result)
    return freq, amplitude

def extract_frequency_bands(freq, amplitude):
    # Rentang frekuensi (dalam Hz)
    delta_band = (0.5, 4.0)
    theta_band = (4.0, 8.0)
    alpha_band = (8.0, 13.0)
    beta_band = (13.0, 30.0)
    gamma_band = (30.0, 40.0)

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
def calculate_power_bands(data, sampling_rate, nperseg, noverlap, nfft):
    eeg_columns = ['TP9', 'AF7', 'AF8', 'TP10']

    delta_power = np.zeros(len(eeg_columns))
    theta_power  # Sesuaikan dengan kebutuhan Anda = np.zeros(len(eeg_columns))
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
    
    # Menampilkan visualisasi
    plot_power_bands('Total', delta_power_db, theta_power_db, alpha_power_db, beta_power_db, gamma_power_db)

    return delta_power, theta_power, alpha_power, beta_power, gamma_power

def plot_power_bands(channel, delta, theta, alpha, beta, gamma):
    labels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    powers = [delta, theta, alpha, beta, gamma]
    x = np.arange(len(labels))
    
    plt.figure(figsize=(10, 6))
    plt.bar(x, powers, width=0.5)
    plt.xlabel('Rentang Frekuensi')
    plt.ylabel('Power')
    plt.title(f'Power Bands untuk Kanal {channel}')
    plt.xticks(x, labels)
    plt.show()

# Memanggil fungsi untuk menghitung power bands
calculate_power_bands(data, sampling_rate, nperseg, noverlap, nfft)

def plot_frequency_comparison(freq, amplitude, channel_name):
    delta_band = (0.5, 4.0)
    theta_band = (4.0, 8.0)
    alpha_band = (8.0, 13.0)
    beta_band = (13.0, 30.0)
    gamma_band = (30.0, 40.0)  # Sesuaikan dengan kebutuhan Anda

    plt.figure(figsize=(12, 6))
    plt.title(f'{channel_name} Frequency Bands')
    
    # Delta
    delta_amplitude = amplitude[(freq >= delta_band[0]) & (freq <= delta_band[1])]
    plt.plot(freq[(freq >= delta_band[0]) & (freq <= delta_band[1])], delta_amplitude, label='Delta')
    
    # Theta
    theta_amplitude = amplitude[(freq >= theta_band[0]) & (freq <= theta_band[1])]
    plt.plot(freq[(freq >= theta_band[0]) & (freq <= theta_band[1])], theta_amplitude, label='Theta')
    
    # Alpha
    alpha_amplitude = amplitude[(freq >= alpha_band[0]) & (freq <= alpha_band[1])]
    plt.plot(freq[(freq >= alpha_band[0]) & (freq <= alpha_band[1])], alpha_amplitude, label='Alpha')
    
    # Beta
    beta_amplitude = amplitude[(freq >= beta_band[0]) & (freq <= beta_band[1])]
    plt.plot(freq[(freq >= beta_band[0]) & (freq <= beta_band[1])], beta_amplitude, label='Beta')
    
    # Gamma
    gamma_amplitude = amplitude[(freq >= gamma_band[0]) & (freq <= gamma_band[1])]
    plt.plot(freq[(freq >= gamma_band[0]) & (freq <= gamma_band[1])], gamma_amplitude, label='Gamma')
    
    plt.xlabel('Frekuensi (Hz)')
    plt.ylabel('Amplitudo')
    plt.xlim(0, 40)  # Batasi jangkauan frekuensi
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot untuk TP9
freq_tp9, amplitude_tp9 = fft_feature_extraction(tp9_data, sampling_rate)
plot_frequency_comparison(freq_tp9, amplitude_tp9, 'TP9')

# Plot untuk AF7
freq_af7, amplitude_af7 = fft_feature_extraction(af7_data, sampling_rate)
plot_frequency_comparison(freq_af7, amplitude_af7, 'AF7')

# Plot untuk AF8
freq_af8, amplitude_af8 = fft_feature_extraction(af8_data, sampling_rate)
plot_frequency_comparison(freq_af8, amplitude_af8, 'AF8')

# Plot untuk TP10
freq_tp10, amplitude_tp10 = fft_feature_extraction(tp10_data, sampling_rate)
plot_frequency_comparison(freq_tp10, amplitude_tp10, 'TP10')

# #ABSOLUT NILAI FFT BIASA
# # Data Delta, Theta, Alpha, Beta, dan Gamma dari berbagai kanal
# delta_values = [delta_tp9, delta_tp10, delta_af7, delta_af8]
# theta_values = [theta_tp9, theta_tp10, theta_af7, theta_af8]
# alpha_values = [alpha_tp9, alpha_tp10, alpha_af7, alpha_af8]
# beta_values = [beta_tp9, beta_tp10, beta_af7, beta_af8]
# gamma_values = [gamma_tp9, gamma_tp10, gamma_af7, gamma_af8]

# # Menjumlahkan nilai dari berbagai kanal
# total_delta = np.sum(delta_values)
# total_theta = np.sum(theta_values)
# total_alpha = np.sum(alpha_values)
# total_beta = np.sum(beta_values)
# total_gamma = np.sum(gamma_values)

# delta_B = (np.abs(total_delta))
# theta_B = (np.abs(total_theta))
# alpha_B = (np.abs(total_alpha))
# beta_B = (np.abs(total_beta))
# gamma_B = (np.abs(total_gamma))

# # Menampilkan hasil penjumlahan
# print(f'Total Delta: {delta_B}')
# print(f'Total Theta: {theta_B}')
# print(f'Total Alpha: {alpha_B}')
# print(f'Total Beta: {beta_B}')
# print(f'Total Gamma: {gamma_B}')

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

# Tampilkan mean dari masing-masing power
print("Mean Delta Power:", mean_delta_power)
print("Mean Theta Power:", mean_theta_power)
print("Mean Alpha Power:", mean_alpha_power)
print("Mean Beta Power:", mean_beta_power)
print("Mean Gamma Power:", mean_gamma_power)

# Plot rata-rata Delta, Theta, Alpha, Beta, dan Gamma
def plot_mean_power(mean_delta, mean_theta, mean_alpha, mean_beta, mean_gamma):
    labels = ['TP9', 'TP10', 'AF7', 'AF8']
    x = np.arange(len(labels))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - 2 * width, mean_delta, width, label='Delta')
    rects2 = ax.bar(x - width, mean_theta, width, label='Theta')
    rects3 = ax.bar(x, mean_alpha, width, label='Alpha')  # Fix this line
    rects4 = ax.bar(x + width, mean_beta, width, label='Beta')
    rects5 = ax.bar(x + 2 * width, mean_gamma, width, label='Gamma')

    ax.set_xlabel('Kanal')
    ax.set_ylabel('Rata-rata Power')
    ax.set_title('Rata-rata Power dalam Berbagai Kanal dan Rentang Frekuensi')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.show()

# Plot rata-rata Delta, Theta, Alpha, Beta, dan Gamma
plot_mean_power(mean_delta_power, mean_theta_power, mean_alpha_power, mean_beta_power, mean_gamma_power)

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

# Plot nilai rata-rata Delta, Theta, Alpha, Beta, dan Gamma untuk semua kanal
def plot_all_mean_power(all_mean_delta, all_mean_theta, all_mean_alpha, all_mean_beta, all_mean_gamma):
    labels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    x = np.arange(len(labels))
    mean_powers = [all_mean_delta, all_mean_theta, all_mean_alpha, all_mean_beta, all_mean_gamma]
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects = ax.bar(x, mean_powers, width)

    ax.set_xlabel('Rentang Frekuensi')
    ax.set_ylabel('Rata-rata Power')
    ax.set_title('Rata-rata Power dalam Berbagai Rentang Frekuensi untuk Semua Kanal')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    plt.show()

# Plot nilai rata-rata Delta, Theta, Alpha, Beta, dan Gamma untuk semua kanal
plot_all_mean_power(all_mean_delta_power, all_mean_theta_power, all_mean_alpha_power, all_mean_beta_power, all_mean_gamma_power)