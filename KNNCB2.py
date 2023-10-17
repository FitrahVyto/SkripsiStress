import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Membaca data dari file CSV (sesuaikan dengan nama dan format file Anda)
data = pd.read_csv('Klasifikasi.csv')

# Pisahkan fitur dan label
X = data[['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']].values
y = data['Label'].values

# Menggunakan LabelEncoder untuk mengonversi label menjadi angka
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Visualisasi data jika perlu
plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=y_encoded, cmap='viridis', marker='*', s=100, edgecolors='black')
plt.show()

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, random_state=0)

# Buat model KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Lakukan prediksi
y_pred = knn.predict(X_test)

# Evaluasi akurasi
accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy:", accuracy)

# Visualisasi hasil prediksi jika perlu
plt.figure(figsize=(10, 10))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', marker='*', s=100, edgecolors='black')
plt.title("Predicted values with k=5", fontsize=20)

plt.show()
