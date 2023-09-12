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

# Transform to numpy array
audio_array = audio.get_array_of_samples().tolist()
audio_array = np.array(audio_array)


# Time constants
T_p = 0.02 # s
# Calculate samples devided by time to get sample rate
sample_rate = int(len(audio_array)/audio.duration_seconds)
# Maximum frequency due to nyquist
f_max = sample_rate/2
# Timer per sample
t_p = 1/sample_rate


# Convolute signal with a window of 45 samples centered at sample 23
kernal = np.array([x[0] for x in cv2.getGaussianKernel(45, 23)])
audio_array_mean = np.convolve(audio_array, kernal, mode='same')
audio_array_without_sync = audio_array - audio_array_mean



# Find the zero crossings of the audio signal
zero_crossings = np.where(np.diff(np.sign(audio_array_mean)))[0]

# # Draw the signal with zero crossings
# plt.plot(((audio_array[5000:15000]-np.mean(audio_array[5000:15000]))/np.std(audio_array)))
# plt.plot(np.diff(np.sign(audio_array_mean))[5000:15000])         
# plt.show()

# How many elements are in each row
kernal = [-1, 1]
zero_crossings_dist = np.abs(np.convolve(zero_crossings, kernal, mode='same')) 
max_dist = np.max(zero_crossings_dist)

# How many columns are in the image
num_rows = int((len(zero_crossings)-1)/2)
print(num_rows)

# Create a blank numpy array
range_img = np.zeros((num_rows,8*max_dist))
# Fill in the image array with data from audio array mean 
for i in range(num_rows):
    start = zero_crossings[2*i+1]
    end = zero_crossings[2*i+2]
    safty_margin = 0
    range_img[i,0:end-start] = -audio_array_without_sync[start:end]#-(audio_array[start+safty_margin:end-safty_margin]-mean)/std
    # Create a time array


# # Show range image
# plt.imshow(range_img)
# plt.show()










# Flatten range image
# range_img = range_img.flatten()
# # Plot the range image
# plt.plot(range_img[5000:15000])
# plt.show()


#freq = np.fft.fftfreq(range_img.shape[1], d=t_p)
    
# Take fft of each row of the image
img_fft = np.fft.ifft(range_img, axis=1)



# Take the absolute value of the fft
img_fft = np.abs(img_fft)

# Take the log of the fft
img_fft = 20*np.log(img_fft)

# Scale each row of the fft to 0-255
img_fft = img_fft - np.min(img_fft, axis=1)[:,None]
img_fft = img_fft / np.max(img_fft, axis=1)[:,None]

# Scale the fft to 0-255 and convert to uint8
img_fft = (img_fft*255).astype(np.uint8)

# Color encode the fft 
img_fft = cv2.applyColorMap(img_fft, cv2.COLORMAP_PLASMA)
 
# Take real part up to nyquist frequency
img_fft = img_fft[:,int(img_fft.shape[1]/2):-int(img_fft.shape[1]/4)]

# # Show the image of the fft using matplotlib
plt.imshow(img_fft)
plt.xlabel('Frequency [Hz]')
plt.show()