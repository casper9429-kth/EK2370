# This code intents to show how to use pydub to read m4a file
# And transform it to a numpy array



from pydub import AudioSegment
from pydub.utils import mediainfo
import matplotlib.pyplot as plt
import numpy as np

path = "/home/postenoden/KTH/Build Your Own Radar System/Project/EK2370/data_cots/Velocity_Test_File.m4a"

# Read m4a file
audio = AudioSegment.from_file(path, format="m4a")

# Transform to numpy array
audio_array = audio.get_array_of_samples()
audio_array = np.array(audio_array)

# Calculate the mean and standard deviation of the audio signal
mean = audio_array.mean()
std = audio_array.std()

# Normalize the audio signal
audio_array = (audio_array - mean) / std

# Take 1000 samples of the audio signal 
#audio_array = audio_array[10000:11000]

# Show a plot of the audio signal
plt.plot(audio_array)
plt.show()