import pandas as pd
import joblib
import time
import I2C_LCD_driver
mylcd = I2C_LCD_driver.lcd()
# Start timer
start_time = time.perf_counter()

# Baca data CSV tanpa label
data_to_classify = pd.read_csv("/home/pifitrah/SkripsiVyto/SkripsiStress/T6_NS_EF.csv")  # Ganti dengan nama file yang sesuai

# Muat model KNN yang sudah disimpan
knn = joblib.load('/home/pifitrah/SkripsiVyto/SkripsiStress/fix_knn_model.joblib')

# Lakukan prediksi pada data tanpa label
predictions = knn.predict(data_to_classify)

# Hasil prediksi
print("Hasil prediksi:")
print(predictions)

mylcd.lcd_display_string(" Suspect Result", 1)
mylcd.lcd_display_string(" ".join(predictions), 2, 2)

# End timer
end_time = time.perf_counter()

# Calculate elapsed time
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)