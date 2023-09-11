# This code intents to show how to use pydub to read m4a file
# And transform it to a numpy array



from pydub import AudioSegment
from pydub.utils import mediainfo
import matplotlib.pyplot as plt
import numpy as np

path = "data_cots/Velocity_Test_File.m4a"

# Read m4a file
audio = AudioSegment.from_file(path, format="m4a")

# Transform to numpy array
audio_array = audio.get_array_of_samples()
audio_array = np.array(audio_array)

# Add 4x size of audio_array of zeros to the end of the array
audio_array_padded = np.append(audio_array, np.zeros(len(audio_array)*4))



# Calculate the mean and standard deviation of the audio signal
# mean = audio_array.mean()
# std = audio_array.std()

# Normalize the audio signal
# audio_array = (audio_array - mean) / std

# Peforrm FFT on padded and not padded audio array with higher resolution
fft = np.fft.fft(audio_array)
#fft = np.fft.fft(audio_array)
fft_padded = np.fft.fft(audio_array_padded)

# Shift the FFT
fft = np.fft.fftshift(fft)
fft_padded = np.fft.fftshift(fft_padded)


# Plot the FFT and FFT padded side by side
plt.subplot(1,2,1)
plt.plot(np.abs(fft))
plt.title("FFT")
plt.subplot(1,2,2)
plt.plot(np.abs(fft_padded))
plt.title("FFT Padded")
plt.show()

