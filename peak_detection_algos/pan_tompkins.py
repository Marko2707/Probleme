import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import numpy as np
from scipy.signal import find_peaks, butter, filtfilt

def pan_tompkins_filter(ecg_signal, sampling_rate):
    # Bandpass-Filter mit Passband von 0.5–50 Hz
    filtered_signal = bandpass_filter(ecg_signal, sampling_rate, lowcut=0.5, highcut=49.9)

    # Moving Average Filter mit N-Punkten (zum Beispiel N=5)
    N = 5
    filtered_signal = moving_average_filter(filtered_signal, N)

    return filtered_signal

def bandpass_filter(signal, sampling_rate, lowcut, highcut):
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(1, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal

def moving_average_filter(signal, N):
    return np.convolve(signal, np.ones(N)/N, mode='valid')

def process_and_plot_ecg(record_path, sampling_rate = 100):
    # Laden Sie das WFDB-Signal
    record_name = record_path + "00001_lr"

    # Verwenden Sie wfdb.rdsamp, um den Pfad und das EKG-Signal zu erhalten
    record_dat, record_hea = wfdb.rdsamp(record_name)

    # Extrahieren Sie das EKG-Signal aus dem WFDB-Aufzeichnung
    #: == Alle zeitlichen Werte, 0 == Erste Ableitung 
    ecg_signal = record_dat[:, 0]  # Nehmen Sie das Signal der ersten Ableitung


    # Pan-Tompkins-Algorithmus anwenden und R-peaks erhalten
    peaks, _ = find_peaks(pan_tompkins_filter(ecg_signal, sampling_rate = 100), height=0)

    time_vector = np.arange(0, len(ecg_signal)) / sampling_rate
    # Original-EKG-Signal und gefiltertes Signal plotten
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_vector, ecg_signal)
    plt.title('Original EKG Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(pan_tompkins_filter(ecg_signal, sampling_rate))
    plt.plot(peaks, ecg_signal, "*", color="red")
    plt.title('Verarbeitetes EKG Signal mit R-peaks')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

def process_wfdb_record(record_path, output_path):
    # Laden Sie das WFDB-Signal
    record_name = record_path + "00001_lr"

    # Verwenden Sie wfdb.rdsamp, um den Pfad und das EKG-Signal zu erhalten
    record_dat, record_hea = wfdb.rdsamp(record_name)

    # Extrahieren Sie das EKG-Signal aus dem WFDB-Aufzeichnung
    #: == Alle zeitlichen Werte, 0 == Erste Ableitung 
    ecg_signal = record_dat[:, 0]  # Nehmen Sie das Signal der ersten Ableitung
    plot_ecg(ecg_signal, 100)

    print(pan_tompkins_algorithm(ecg_signal, 100))

    
def plot_ecg(ecg_signal, sampling_rate):
    # Zeitvektor erstellen
    time_vector = np.arange(0, len(ecg_signal)) / sampling_rate

    # Plot des EKG-Signals
    plt.plot(time_vector, ecg_signal)
    plt.title('EKG-Signal')
    plt.xlabel('Zeit (s)')
    plt.ylabel('Amplitude')
    plt.show()


def main():
    # Pfad zur WFDB-Aufzeichnung und Ausgabeordner
    folder_path = "C:/Users/marko/Desktop/bachelor_arbeit/code/records100/00000/"
    output_folder = "C:/Users/marko/Desktop/Pan-Tompkin/Output/"
    process_and_plot_ecg(folder_path, 100)
    #process_wfdb_record(folder_path, output_folder) 

main()