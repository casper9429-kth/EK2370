# This code intents to show how to use pydub to read m4a file
# And transform it to a numpy array
from pydub import AudioSegment
from pydub.utils import mediainfo
import matplotlib.pyplot as plt
import numpy as np
import cv2 

path = "data_cots/Range_Test_File.m4a"

# Read m4a file
audio = AudioSegment.from_file(path, format="m4a")


# Get the different channels 
audio_channels = audio.split_to_mono()
audio_array = []
for channel in audio_channels:
    audio_array.append(channel.get_array_of_samples().tolist())
audio_array = np.array(audio_array)

arr1 = audio_array[0,:]
arr2 = audio_array[1,:]

# Plot the two channels
sync = np.zeros_like(arr2)
sync[arr2>0] = 1
sync[arr2<0] = -1
# Plot the sync signal
#plt.plot(sync[0:5000])
#plt.show()

# Get the zero crossings of the sync signal
zero_crossings = np.where(np.diff(np.sign(sync)))[0]


# Get sampling frequency
sample_rate = int(len(arr1)/audio.duration_seconds)

# Each Chirp is 20 ms long
Tp = 0.02 # s

# Each Chirp is N samples long
N = int(Tp*sample_rate)


pass