from datetime import datetime
from pythonosc import dispatcher
from pythonosc import osc_server
import csv
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from sklearn.decomposition import FastICA

ip = "10.34.3.220"
port = 5000

#Test
# Inisialisasi variabel-variabel untuk merekam data EEG
eeg_data = []
record_duration = 60 # Durasi perekaman dalam detik
record_started = False
server = None  # Variabel untuk menyimpan objek server

# Fungsi untuk menghandle pesan EEG
def eeg_handler(address: str, *args):
    global eeg_data, record_started

    # Cek apakah perekaman sudah dimulai
    if not record_started:
        record_started = True
        # Mulai thread timer untuk mengakhiri perekaman setelah 60 detik
        threading.Timer(record_duration, stop_recording).start()

    # Simpan data EEG ke dalam variabel
    eeg_data.append([datetime.now()] + list(args))

    # Cetak data EEG ke konsol
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), *args)

# Fungsi untuk mengakhiri perekaman dan menyimpan data ke dalam file CSV
def stop_recording():
    global eeg_data, record_started, server

    if record_started:
        # Simpan data ke dalam file CSV
        with open("eegcoba_SSTRESS.csv", "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            # Tulis header
            csv_writer.writerow(["Timestamp", "TP9", "AF7", "AF8", "TP10", "AUX"])
            # Tulis data
            csv_writer.writerows(eeg_data)

        print("Data EEG telah disimpan dalam eeg_SSTRESS.csv")
        eeg_data = []
        record_started = False
        # Hentikan server OSC
        if server is not None:
            server.shutdown()
            server.server_close()

if __name__ == "__main__":
    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/muse/eeg", eeg_handler)

    server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
    print("Listening on UDP port " + str(port))
    server.serve_forever()

# Load the recorded EEG data
eeg = pd.read_csv('eegcoba_SSTRESS.csv')
eeg.head()

eeg.isnull().sum()

eeg = eeg.copy()  # Buat salinan DataFrame
timestamps = eeg['Timestamp']
eeg = eeg.drop(['Timestamp', 'AUX'], axis=1)  # Hapus kolom waktu dari salinan DataFrame

ica = FastICA(n_components=4, random_state=0, tol=0.0001)
comps = ica.fit_transform(eeg)

comps_restored = comps.copy()
comps_restored[:,[2,3]] = 0# set artefact components to zero
restored = ica.inverse_transform(comps_restored)

eeg['Timestamp'] = timestamps

# Buat DataFrame baru dengan data yang sudah di-restorasi
restored_df = pd.DataFrame({
    'Timestamp': timestamps,  # Tambahkan kembali kolom Timestamp
    # 'TP9_pre': eeg['TP9'],
    'TP9': restored[:, 0],
    # 'AF7_pre': eeg['AF7'],
    'AF7': restored[:, 1],
    # 'AF8_pre': eeg['AF8'],
    'AF8': restored[:, 2],
    # 'TP10_pre': eeg['TP10'],
    'TP10': restored[:, 3],
    # 'ICA_Comp_4': comps[:, 3]  # Ganti nama sesuai kebutuhan
})

# Save the preprocessed data to a new CSV file
restored_df.to_csv('HASILPROSEScsv.csv', index=False)  # Save the preprocessed data to a new CSV file
