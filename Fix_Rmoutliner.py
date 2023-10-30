import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca data EEG dari file CSV
df = pd.read_csv("Fix_ARDI.csv")

# Kolom yang akan dianalisis (meliputi tp9, af7, af8, tp10)
columns = ["TP9", "AF7", "AF8", "TP10"]

# Menentukan threshold Z-score menggunakan mean dan std
threshold = 3

# Mencari dan menghapus outlier
outlier_indices = []

for column in columns:
    data = df[column]
    mean = np.mean(data)
    std = np.std(data)
    
    for i in range(len(data)):
        z = (data[i] - mean) / std
        if abs(z) > threshold:
            outlier_indices.append(i)

# Menghapus baris yang mengandung outlier
df_cleaned = df.drop(outlier_indices)

# Menyimpan data yang telah dibersihkan ke file CSV baru
df_cleaned.to_csv("ARDI_C.csv", index=False)

# Informasi tentang outlier yang dihapus
print("Jumlah outlier yang dihapus menggunakan mean dan std:", len(outlier_indices))
print("Indeks outlier:", outlier_indices)

# Data yang telah dibersihkan tersimpan dalam df_cleaned

# Visualisasi data outlier menggunakan Seaborn
plt.figure(figsize=(12, 6))
plt.suptitle("Outlier Detection using Seaborn")

for i, column in enumerate(columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=df, x=column, color='skyblue')
    plt.title(f'Box Plot for {column}')
    plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Visualisasi data yang telah dibersihkan
plt.figure(figsize=(12, 6))
plt.suptitle("Data Setelah Outlier Dihapus")

for i, column in enumerate(columns, 1):
    plt.subplot(2, 2, i)
    sns.histplot(data=df_cleaned, x=column, kde=True, color='skyblue', bins=20)
    plt.title(f'Histogram for {column}')
    plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for column in columns:
    plt.subplot(2, 2, columns.index(column) + 1)
    plt.scatter(range(len(df)), df[column], c='b', label='Data')
    plt.scatter(outlier_indices, df[column].iloc[outlier_indices], c='r', label='Outlier')
    plt.title(f'Outlier Detection for {column}')
    plt.legend()

plt.tight_layout()
plt.show()

# Visualisasi data dengan pair plot menggunakan Seaborn
sns.set(style="ticks")
sns.pairplot(df_cleaned[columns], height=2, corner=True, diag_kind="kde", markers="o")

plt.suptitle("Pair Plot of EEG Data")
plt.show()

# Visualisasi data dengan heatmap menggunakan Seaborn
correlation_matrix = df_cleaned[columns].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of EEG Data")
plt.show()


# Visualisasi data dengan scatter plot matrix menggunakan Seaborn
sns.set(style="whitegrid")
sns.pairplot(df_cleaned[columns], height=2, markers="o")

plt.suptitle("Scatter Plot Matrix of EEG Data")
plt.show()

