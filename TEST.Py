import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Membaca data dari file CSV yang ingin diprediksi (sesuaikan dengan nama dan format file Anda)
data_to_predict = pd.read_csv('TESTING.csv')

# Pisahkan fitur dari data yang ingin diprediksi
X_to_predict = data_to_predict[['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']].values

# Menggunakan LabelEncoder untuk mengonversi label jika diperlukan
# label_encoder = LabelEncoder()
# Jika label kelasnya sudah dalam format angka, tidak perlu dienkoding ulang

# Memuat model KNN yang telah dilatih sebelumnya
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Lakukan prediksi pada data yang ingin diprediksi
y_pred = knn.predict(X_to_predict)

# Hasil prediksi dapat Anda simpan atau tampilkan
print("Hasil Prediksi:", y_pred)
