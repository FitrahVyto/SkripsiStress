import pandas as pd
import joblib

# Baca data CSV tanpa label
data_to_classify = pd.read_csv("D:\KULIAH\SEMESTER 7 SKRIPSI\Backup\TESTING.csv")  # Ganti dengan nama file yang sesuai

# Muat model KNN yang sudah disimpan
knn = joblib.load('model.joblib')

# Lakukan prediksi pada data tanpa label
predictions = knn.predict(data_to_classify)

# Hasil prediksi
print("Hasil prediksi:")
for prediction in predictions:
    print(prediction)