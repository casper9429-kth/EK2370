import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.utils import mediainfo

def read_data_sync_fs(path):
    """
    Read data, sync and fs from m4a file using pydub
    return: data, sync, fs [numpy array, numpy array, int]
    """
    audio = AudioSegment.from_file(path, format="m4a")
    audio_channels = audio.split_to_mono()
    audio_array = [channel.get_array_of_samples().tolist() for channel in audio_channels]
    audio_array = np.array(audio_array)
    fs = int(mediainfo(path)['sample_rate'])

    # Print some information
    print("Audio channels: ", len(audio_channels))
    print("Audio array shape: ", audio_array.shape)
    print("Audio fs: ", fs)
    data = audio_array[0]
    sync = audio_array[1]
    return data, sync, fs   



# Define the file path to your audio file
file_path = '/home/casper/EK2370/Subgrupp_2/Velocity_Test_File.m4a'

# Read the audio file and get the audio data and sampling rate
# Read the audio file and get the audio data and sampling rate
audio_data,sync, sample_rate = read_data_sync_fs(file_path)

# Invert the audio data
audio_data_inv = -audio_data#audio_data[:, 0]

N = len(audio_data_inv)
T = 1 / sample_rate
Tp = 0.1

# Amount of samples per sweeps
Sample_per_sweep = int(Tp / T)

# This is the number of sweeps
M = N // Sample_per_sweep

# Creating a 2D array
First_array = np.zeros((M - 1, Sample_per_sweep))

# Variable to keep track of each sample
k = 1
velocities = np.linspace(0, sample_rate / 2, 2 * Sample_per_sweep)
velocities = velocities * (3 * 10**8) / (2 * 2.43 * 10**9)

time_array = np.linspace(0, Tp * (M - 1), M - 1)

audio_data_inv = audio_data_inv[:Sample_per_sweep * (M - 1)].reshape((Sample_per_sweep, M - 1))
First_array[:, :Sample_per_sweep] = audio_data_inv.T

# Clutter rejection
First_array[:, :Sample_per_sweep] = First_array[:, :Sample_per_sweep] - np.mean(First_array[:, :Sample_per_sweep])

# FFT
fft_first = 10 * np.log10(np.abs(np.fft.fft(First_array, 5 * Sample_per_sweep, axis=1)))
fft_first = fft_first[:, :Sample_per_sweep * 2]

# Normalization (Norm2)
max_rows = np.max(fft_first, axis=1)
fft_first = fft_first - max_rows[:, np.newaxis]

# Min 5 precentile of fft_first
min_5 = np.percentile(fft_first, 5)
# Max 95 precentile of fft_first
max_95 = np.percentile(fft_first, 95)
# Set all values below min_5 to min_5
# fft_first[fft_first < min_5] = min_5
# # Set all values above max_95 to max_95
# fft_first[fft_first > max_95] = max_95
# # Scale the values between 0 and 1
# fft_first = (fft_first - min_5) / (max_95 - min_5)
# # Scale the values between 0 and 255


# Plotting
plt.imshow(fft_first, extent=(0, 30, time_array[-1], time_array[0]), aspect='auto', vmin=-20, vmax=0)
plt.colorbar(label='Intensity (dB)')
plt.xlabel('Velocities')
plt.ylabel('Time')
plt.title('Spectrogram')
plt.show()
