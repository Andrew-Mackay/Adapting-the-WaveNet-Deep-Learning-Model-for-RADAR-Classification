import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from scipy.signal import butter, freqz, lfilter, spectrogram


# Switches out i for j to ensure python compatibility
def convert_to_complex(complex_string):
    return complex(complex_string[0].replace('i', 'j'))


def nearest_odd_number(x):
    if(np.floor(x) % 2 == 0):
        return int(np.floor(x) + 1)
    else:
        return int(np.floor(x))


def make_spectrograms(df):
    # Grab RADAR settings from top of file
    center_frequency = float(df.iloc[0])
    sweep_time = float(df.iloc[1]) / 1000  # convert to seconds
    number_of_time_samples = float(df.iloc[2])
    bandwidth = float(df.iloc[3])
    sampling_frequency = number_of_time_samples / sweep_time
    record_length = 60
    number_of_chirps = record_length / sweep_time

    # Put data values into an array
    data = df.iloc[4:]
    data = data.apply(convert_to_complex, axis=1)
    data = data.values

    # Reshape into chirps over time
    data_time = np.reshape(data, (int(number_of_chirps),int(number_of_time_samples)))
    data_time = np.rot90(data_time, k=3)

    win = np.ones((int(number_of_time_samples), data_time.shape[1]))
    # Apply fast fourier transform
    fft_applied = np.fft.fft((data_time * win), axis=0)

    # shift fft (swap top and bottom half
    fft_shifted = np.fft.fftshift(fft_applied, axes=(0,))

    # take top half
    data_range = fft_shifted[int(number_of_time_samples / 2):int(number_of_time_samples), :]

    # IIR Notch filter
    x = data_range.shape[1]
    ns = nearest_odd_number(x) - 1

    data_range_MTI = np.zeros((data_range.shape[0], ns), dtype=np.complex128)

    # made a filter remove DC component and very low frequency components
    (b, a) = butter(4, 0.01, btype="high")
    for i in range(data_range.shape[0]):
        data_range_MTI[i, :ns] = lfilter(b, a, data_range[i, :ns], axis=0)

    freq = np.arange(0, ns - 1)

    data_range_MTI = data_range_MTI[1:, :]

    bin_indl = 5
    bin_indu = 25
    time_window_length = 200
    overlap_factor = 0.95
    overlap_length = np.round(time_window_length * overlap_factor)
    pad_factor = 4
    fft_points = pad_factor * time_window_length

    data_spec_MTI2 = 0
    for rbin in range(bin_indl - 1, bin_indu):
        s, f, t = mlab.specgram(data_range_MTI[rbin, :],
                                Fs=1,
                                window=np.hamming(time_window_length),
                                noverlap=overlap_length,
                                NFFT=time_window_length,
                                mode='complex',
                                pad_to=fft_points)

        data_MTI_temp = np.fft.fftshift(s, 1)
        data_spec_MTI2 = data_spec_MTI2 + abs(data_MTI_temp)

    iterations = 5700  # 57 seconds
    window_size = 300  # 3 seconds
    step_size = 10  # 0.1 seconds
    spectrograms = []
    for i in range(0, iterations-step_size, step_size):
        center = int(data_spec_MTI2.shape[0]/2)
        data_spec_small = data_spec_MTI2[(center-150):(center+150), i:(i + window_size)]
        spectrograms.append(data_spec_small)

    return spectrograms
