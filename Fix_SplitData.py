import pandas as pd
import os

# Membaca data EEG dari file CSV
csv_file_path = "Fix_SYAFATIRA.csv"
df = pd.read_csv(csv_file_path)

# Mendefinisikan rentang waktu dan nama file yang sesuai
time_ranges = [("2023-10-21 09:35:03", "2023-10-21 09:40:03", "AA"),
               ("2023-10-21 09:40:03", "2023-10-21 10:10:03", "BB"),
               ("2023-10-21 10:10:03", "2023-10-21 10:18:03", "CC"),
               ("2023-10-21 10:18:03", "2023-10-21 10:25:03", "DD"),
               ("2023-10-21 10:25:03", "2023-10-21 10:26:03", "EE")]

# Nama folder untuk menyimpan potongan data
output_folder = "sliced_data"

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
