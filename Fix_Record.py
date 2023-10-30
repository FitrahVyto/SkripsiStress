from datetime import datetime
from pythonosc import dispatcher
from pythonosc import osc_server
import csv
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

ip = "192.168.196.175"
port = 8080

#Test
# Inisialisasi variabel-variabel untuk merekam data EEG
eeg_data = []
record_duration = 3060 # Durasi perekaman dalam detik
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
    
    if record_started:
    # Simpan data ke dalam file CSV
        with open("YUDISTHIRA.csv", "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            # Tulis header
            csv_writer.writerow(["Timestamp", "TP9", "AF7", "AF8", "TP10", "AUX"])
            # Tulis data
            csv_writer.writerows(eeg_data)

    # Cetak data EEG ke konsol
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), *args)

# Fungsi untuk mengakhiri perekaman dan menyimpan data ke dalam file CSV
def stop_recording():
    global eeg_data, record_started, server

    if record_started:
        print("Data EEG telah disimpan dalam eeg_SSTRESS.csv")
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