import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib

# Baca data CSV yang sudah dilabeli
data = pd.read_csv("/home/pifitrah/SkripsiVyto/SkripsiStress/SIIIII.csv")

# Pisahkan fitur (X) dan label (y)
X = data.drop('Label', axis=1)  # Ubah 'label' menjadi nama kolom label sesuai dengan data Anda
y = data['Label']

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model KNN
k = 7  # Ganti dengan nilai K yang sesuai
knn = KNeighborsClassifier(n_neighbors=k)

# Latih model KNN pada data latih
knn.fit(X_train, y_train)

# Lakukan prediksi pada data uji
y_pred = knn.predict(X_test)

# Evaluasi model
print(classification_report(y_test, y_pred))

# Simpan model ke dalam file menggunakan joblib
joblib.dump(knn, 'knn_model.joblib')