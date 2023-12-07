import pandas as pd
import os

# Membaca data EEG dari file CSV
csv_file_path = "/home/pifitrah/SkripsiVyto/SkripsiStress/DataTesting/BERSIH/Fix_60_Attar_PREPROCESSING.csv"
df = pd.read_csv(csv_file_path)

# 5 MENIT NS
# 38 MENIT HS
# 8 MENIT MS

# Mendefinisikan rentang waktu dan nama file yang sesuai
time_ranges = [("2023-12-07 15:02:45", "2023-12-07 15:07:45", "T5_NS"),# 5 MENIT NS
               ("2023-12-07 15:07:45", "2023-12-07 15:45:45", "T5_HS"),# 38 MENIT HS
               ("2023-12-07 15:45:45", "2023-12-07 15:53:45", "T5_MS")]# 8 MENIT MS

# Nama folder untuk menyimpan potongan data
output_folder = "T5_ATTAR"

# Buat folder jika belum ada
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Konversi kolom timestamp ke dalam tipe data datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Memotong data berdasarkan timestamp dan menyimpan ke file CSV di folder
for start_time, end_time, file_name in time_ranges:
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    
    sliced_df = df[(df['Timestamp'] >= start_time) & (df['Timestamp'] <= end_time)]

    # Simpan subset data ke file CSV di dalam folder dengan nama yang sesuai
    file_path = os.path.join(output_folder, f"{file_name}.csv")
    sliced_df.to_csv(file_path, index=False)
