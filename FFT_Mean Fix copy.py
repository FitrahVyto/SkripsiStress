import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Membaca file CSV
data = pd.read_csv('S.csv')

# Pilih kolom yang berisi data sensor Muse (misalnya, TP9, AF7, AF8, TP10)
tp9_data = data['TP9'].values
af7_data = data['AF7'].values
af8_data = data['AF8'].values
tp10_data = data['TP10'].values

# Frekuensi sampling, sesuaikan dengan data Anda (contoh: 256 Hz)
sampling_rate = 256

# Fungsi untuk melakukan Transformasi Fourier Cepat (FFT) pada data sensor
def fft_feature_extraction(data, sampling_rate):
    n = len(data)
    fft_result = np.fft.fft(data)
    freq = np.fft.fftfreq(n, 1 / sampling_rate)
    amplitude = np.abs(fft_result)
    return freq, amplitude

# Fungsi untuk mengekstraksi amplitudo dalam berbagai rentang frekuensi
def extract_frequency_bands(freq, amplitude):
    # Rentang frekuensi (dalam Hz)
    delta_band = (0.5, 4.0)
    theta_band = (4.0, 8.0)
    alpha_band = (8.0, 13.0)
    beta_band = (13.0, 30.0)
    gamma_band = (30.0, 40.0)  # Sesuaikan dengan kebutuhan Anda

    # Ekstraksi amplitudo dalam rentang frekuensi
    delta_power = np.trapz(amplitude[(freq >= delta_band[0]) & (freq <= delta_band[1])])
    theta_power = np.trapz(amplitude[(freq >= theta_band[0]) & (freq <= theta_band[1])])
    alpha_power = np.trapz(amplitude[(freq >= alpha_band[0]) & (freq <= alpha_band[1])])
    beta_power = np.trapz(amplitude[(freq >= beta_band[0]) & (freq <= beta_band[1])])
    gamma_power = np.trapz(amplitude[(freq >= gamma_band[0]) & (freq <= gamma_band[1])])

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

# Menampilkan hasil ekstraksi fitur untuk masing-masing kanal

# Menampilkan hasil ekstraksi fitur TP9
print("Delta Power (TP9):", delta_tp9)
print("Theta Power (TP9):", theta_tp9)
print("Alpha Power (TP9):", alpha_tp9)
print("Beta Power (TP9):", beta_tp9)
print("Gamma Power (TP9):", gamma_tp9)

# Menampilkan hasil ekstraksi fitur AF7
print("Delta Power (AF7):", delta_af7)
print("Theta Power (AF7):", theta_af7)
print("Alpha Power (AF7):", alpha_af7)
print("Beta Power (AF7):", beta_af7)
print("Gamma Power (AF7):", gamma_af7)

# Menampilkan hasil ekstraksi fitur AF8
print("Delta Power (AF8):", delta_af8)
print("Theta Power (AF8):", theta_af8)
print("Alpha Power (AF8):", alpha_af8)
print("Beta Power (AF8):", beta_af8)
print("Gamma Power (AF8):", gamma_af8)

# Menampilkan hasil ekstraksi fitur TP9
print("Delta Power (TP10):", delta_tp10)
print("Theta Power (TP10):", theta_tp10)
print("Alpha Power (TP10):", alpha_tp10)
print("Beta Power (TP10):", beta_tp10)
print("Gamma Power (TP10):", gamma_tp10)

# Plot sinyal frekuensi untuk Delta, Theta, Alpha, dan Beta dalam satu gambar
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

# Plot untuk setiap kanal
plot_frequency_comparison(freq_tp9, amplitude_tp9, 'TP9')
plot_frequency_comparison(freq_af7, amplitude_af7, 'AF7')
plot_frequency_comparison(freq_af8, amplitude_af8, 'AF8')
plot_frequency_comparison(freq_tp10, amplitude_tp10, 'TP10')

# Data delta, theta, alpha, beta, gamma untuk semua kanal
delta_values = [delta_tp9, delta_tp10, delta_af7, delta_af8]
theta_values = [theta_tp9, theta_tp10, theta_af7, theta_af8]
alpha_values = [alpha_tp9, alpha_tp10, alpha_af7, alpha_af8]
beta_values = [beta_tp9, beta_tp10, beta_af7, beta_af8]
gamma_values = [gamma_tp9, gamma_tp10, gamma_af7, gamma_af8]

# Hitung rata-rata Delta, Theta, Alpha, Beta, dan Gamma power untuk masing-masing kanal
mean_delta_power = [np.mean(delta_values[i]) for i in range(len(delta_values))]
mean_theta_power = [np.mean(theta_values[i]) for i in range(len(theta_values))]
mean_alpha_power = [np.mean(alpha_values[i]) for i in range(len(alpha_values))]
mean_beta_power = [np.mean(beta_values[i]) for i in range(len(beta_values))]
mean_gamma_power = [np.mean(gamma_values[i]) for i in range(len(gamma_values))]

# Tampilkan rata-rata power
print("Mean Delta Power:", mean_delta_power)
print("Mean Theta Power:", mean_theta_power)
print("Mean Alpha Power:", mean_alpha_power)
print("Mean Beta Power:", mean_beta_power)
print("Mean Gamma Power:", mean_gamma_power)

# Plot nilai rata-rata Delta, Theta, Alpha, Beta, dan Gamma untuk semua kanal
def plot_mean_power(mean_delta, mean_theta, mean_alpha, mean_beta, mean_gamma):
    labels = ['TP9', 'TP10', 'AF7', 'AF8']
    x = np.arange(len(labels))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - 2 * width, mean_delta, width, label='Delta')
    rects2 = ax.bar(x - width, mean_theta, width, label='Theta')
    rects3 = ax.bar(x, mean_alpha, width, label='Alpha')
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
print("All Mean Delta Power:", all_mean_delta_power)
print("All Mean Theta Power:", all_mean_theta_power)
print("All Mean Alpha Power:", all_mean_alpha_power)
print("All Mean Beta Power:", all_mean_beta_power)
print("All Mean Gamma Power:", all_mean_gamma_power)

data_dict = {
    'Delta': [all_mean_delta_power],
    'Theta': [all_mean_theta_power],
    'Alpha': [all_mean_alpha_power],
    'Beta': [all_mean_beta_power],
    'Gamma': [all_mean_gamma_power],
}

df = pd.DataFrame(data_dict)

# Menyimpan DataFrame ke dalam file CSV
df.to_csv('HasilS.csv', index=False)
print("Berhasil disimpan dalam CSV")

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
