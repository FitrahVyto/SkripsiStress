import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt 

# Membaca dataset yang berisi nilai Alpha, Beta, Delta, Theta, Gamma, dan label (stres atau tidak stres)
data = pd.read_csv('Klasifikasi.csv')  # Gantilah 'dataset.csv' dengan nama file dataset Anda

# Memisahkan fitur (Alpha, Beta, Delta, Theta, Gamma) dan label (stres atau tidak stres)
X = data[['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']].values
y = data['Label'].values

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi fitur menggunakan StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Inisialisasi model KNN dengan nilai K yang sesuai (misalnya, K=3)
knn = KNeighborsClassifier(n_neighbors=5)

# Melatih model KNN pada data latih
knn.fit(X_train, y_train)

# Membuat prediksi menggunakan model yang telah dilatih
y_pred = knn.predict(X_test)

# Mengukur akurasi klasifikasi
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi Klasifikasi:", accuracy)

# Menampilkan laporan klasifikasi
classification_report_result = classification_report(y_test, y_pred)
print("Laporan Klasifikasi:\n", classification_report_result)

# Menampilkan matriks konfusi
confusion_matrix_result = confusion_matrix(y_test, y_pred)
print("Matriks Konfusi:\n", confusion_matrix_result)


# contoh: 
# Alpha,Beta,Delta,Theta,Gamma,Label
# 0.25,0.45,0.10,0.05,0.15,Stress
# 0.30,0.40,0.12,0.06,0.18,Stress
# 0.20,0.35,0.08,0.04,0.12,Not Stress
# 0.22,0.38,0.09,0.04,0.14,Not Stress
# 0.28,0.42,0.11,0.05,0.16,Stress
# 0.18,0.33,0.07,0.03,0.10,Not Stress
# 0.24,0.37,0.10,0.05,0.14,Stress
# 0.26,0.39,0.10,0.05,0.16,Stress
# 0.21,0.36,0.08,0.04,0.11,Not Stress
# 0.23,0.40,0.09,0.04,0.13,Not Stress
