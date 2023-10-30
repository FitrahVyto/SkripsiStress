import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca data EEG dari file CSV
df = pd.read_csv("ARDI_C.csv")

# Kolom yang akan dianalisis (meliputi tp9, af7, af8, tp10)
columns = ["TP9", "AF7", "AF8", "TP10"]

# Visualisasi data dengan satuan waktu (asumsi frekuensi sampling 128 Hz)
plt.figure(figsize=(12, 6))
plt.suptitle("Visualisasi Data EEG dengan Satuan Waktu")

# Menambahkan kolom waktu berdasarkan frekuensi sampling
fs = 256  # Frekuensi sampling (Hz)
df['Time'] = df.index / fs

for i, column in enumerate(columns, 1):
    plt.subplot(2, 2, i)
    sns.lineplot(data=df, x='Time', y=column)
    plt.title(f'{column} vs. Time')
    plt.xlabel("Waktu (detik)")
    plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Visualisasi data dengan satuan sample/baris
plt.figure(figsize=(12, 6))
plt.suptitle("Visualisasi Data EEG dengan Satuan Sample/Baris")

for i, column in enumerate(columns, 1):
    plt.subplot(2, 2, i)
    sns.lineplot(data=df, x=df.index, y=column)
    plt.title(f'{column} vs. Sample/Baris')
    plt.xlabel("Sample/Baris")
    plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
