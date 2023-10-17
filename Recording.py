from datetime import datetime
from pythonosc import dispatcher
from pythonosc import osc_server

ip = "10.34.3.220"
port = 5000

# Fungsi handler untuk EEG
def eeg_handler(address: str, *args):
    dateTimeObj = datetime.now()
    printStr = dateTimeObj.strftime("%Y-%m-%d %H:%M:%S.%f")
    for arg in args:
        printStr += "," + str(arg)
    print(printStr)

# Fungsi handler untuk data absolut (delta, theta, alpha, beta, gamma)
def abs_handler(address: str, value, index):
    dateTimeObj = datetime.now()
    printStr = dateTimeObj.strftime("%Y-%m-%d %H:%M:%S.%f")
    printStr += f",{address},{index},{value}"
    print(printStr)

if __name__ == "__main__":
    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/muse/eeg", eeg_handler)
    dispatcher.map("/muse/elements/delta_absolute", abs_handler, 0)
    dispatcher.map("/muse/elements/theta_absolute", abs_handler, 1)
    dispatcher.map("/muse/elements/alpha_absolute", abs_handler, 2)
    dispatcher.map("/muse/elements/beta_absolute", abs_handler, 3)
    dispatcher.map("/muse/elements/gamma_absolute", abs_handler, 4)

    server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
    print("Listening on UDP port " + str(port))
    server.serve_forever()

    # dispatcher.map("/muse/elements/delta_absolute", eeg_handler)
    # dispatcher.map("/muse/elements/theta_absolute", eeg_handler)
    # dispatcher.map("/muse/elements/alpha_absolute", eeg_handler)
    # dispatcher.map("/muse/elements/beta_absolute", eeg_handler)
    # dispatcher.map("/muse/elements/gamma_absolute", eeg_handler)