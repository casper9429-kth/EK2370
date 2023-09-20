# This code intents to show how to use pydub to read m4a file
# And transform it to a numpy array



from pydub import AudioSegment
from pydub.utils import mediainfo
import matplotlib.pyplot as plt
import numpy as np

path = "data_cots/Range_Test_File.m4a"

# Read m4a file
audio = AudioSegment.from_file(path, format="m4a")
# 

# Transform to numpy array
audio_array_in = audio.get_array_of_samples().tolist()
audio_array = np.array(audio_array_in).copy()

# Take absolut value
audio_array_abs = audio_array
# Convolut with a gussian kernel
audio_array_conv = np.convolve(audio_array_abs, np.ones(100)/100, mode='same')
# Take derivative
audio_array_diff = np.diff(audio_array_conv)




plt.plot(audio_array_diff[5000:10000])
plt.show()