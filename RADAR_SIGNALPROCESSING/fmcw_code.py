# This code intents to show how to use pydub to read m4a file
# And transform it to a numpy array
from pydub import AudioSegment
from pydub.utils import mediainfo
import matplotlib.pyplot as plt
import numpy as np
import cv2

def main():
    # Define Path
    path = "data_cots/Range_Test_File.m4a"
    # Get data, sync and fs
    data,sync,fs = read_data_sync_fs(path)
    # Constants
    c = 3e8;                # speed of light [m/s]
    Ts = 1/fs;              # sampling time [s]
    Tp = 0.02;              # pulse width [s]
    fc = 2.43e9;            # center frequency [hz]
    Nsamples = int(Tp*fs);       # number of samples per puls
    BW = (2.495e9 - 2.408e9); # bandwidth of FMCW [Hz]
    padding_constant = 4;   # padding constant for the FFT


    # Extract upchirp indicies, it is where sync goes from negative to positive
    upchirp_indicies = extract_upchrip_indicies(sync)
    
    # Cancel out none upchirp samples
    data = upchrip_pass_filter(data,sync)
    
    # Invert data to get the correct phase
    data = data*-1 

    # Create upchirp matrix
    upchirp_matrix = create_upchirp_matrix(data,upchirp_indicies,Nsamples)
    
    # Peform MS clutter removal
    upchirp_matrix = MS_Clutter_removal(upchirp_matrix)
    
    # Perform MTI
    upchirp_matrix = two_pulse_MTI(upchirp_matrix)

    # Perform 2D FFT
    upchirp_matrix = ifft_with_filtering(upchirp_matrix,padding_constant,Nsamples)
    min_X = np.min(upchirp_matrix)
    max_X = np.max(upchirp_matrix)
    upchirp_matrix = (upchirp_matrix - min_X)/(max_X - min_X)*255

    # Convert to uint8
    upchirp_matrix = upchirp_matrix.astype(np.uint8)
    
    # # Make contrast better
    # upchirp_matrix = cv2.equalizeHist(upchirp_matrix)
    
    # upchirp_matrix = cv2.cvtColor(upchirp_matrix,cv2.COLOR_GRAY2RGB)
    # # Find edges using canny
    # edges = cv2.Canny(upchirp_matrix,100,200)
    # # Peform adaptive thresholding

    
    # Color map
    upchirp_matrix = cv2.applyColorMap(upchirp_matrix, cv2.COLORMAP_JET)
    
    
    # Show image
    cv2.imshow("Image",upchirp_matrix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def ifft_with_filtering(upchirp_matrix,Padding,Nsamples):
    """
    Perform ifft along rows and filter 
    """
    # Ifft
    ifft = np.fft.ifft(upchirp_matrix,axis=1,n=Padding*Nsamples)
    
    # Absolute value
    ifft = np.abs(ifft)

    # Cut away padding
    ifft = ifft[:,:Nsamples]

    # Convolute with a aggressive gaussian filter
    kernel = cv2.getGaussianKernel(30,30)
    ifft = cv2.filter2D(ifft,-1,kernel)


    # Find 5 prcentile and 95 percentile
    min_X = np.percentile(ifft, 1)
    #max_X = np.percentile(ifft, 99.999)
    max_X = np.percentile(ifft, 99.9)

    # Set values below 5 percentile to 5 percentile
    ifft[ifft < min_X] = min_X
    # Set values above 95 percentile to 95 percentile
    ifft[ifft > max_X] = max_X
    
    
    # Logarithmic scale
    ifft = np.log10(ifft)
    
    
    # Scale min and max to 0 and 255
    return ifft
    

def two_pulse_MTI(upchirp_matrix):    
    ### Subtract the previous chirp(row) from the current chirp (row)
    S_2pulse = np.zeros_like(upchirp_matrix)
    S_2pulse[1:,:] = upchirp_matrix[1:,:] - upchirp_matrix[:-1,:] 
    return S_2pulse

def MS_Clutter_removal(upchirp_matrix):
    mean = np.mean(upchirp_matrix,axis=0)
    upchirp_matrix = upchirp_matrix - mean
    return upchirp_matrix

def create_upchirp_matrix(data,upchirp_indicies,Nsamples):
    """
    Create a matrix with all upchirps
    """
    # Create upchirp matrix
    upchirp_matrix = np.zeros((len(upchirp_indicies),Nsamples))
    for i in range(len(upchirp_indicies)):
        # Check if upchirp is within the data
        if upchirp_indicies[i]+Nsamples > len(data):
            print("Upchirp is outside data")
            new_upchirp_matrix = np.zeros((i,Nsamples))
            new_upchirp_matrix[:,:] = upchirp_matrix[:i,:]
            upchirp_matrix = new_upchirp_matrix
            break
        upchirp_matrix[i,:] = data[upchirp_indicies[i]:upchirp_indicies[i]+Nsamples]
    return upchirp_matrix

def upchrip_pass_filter(data,sync):
    """
    Set none upchirp samples to zero
    """
    upchrip_mask = np.zeros_like(sync)
    upchrip_mask[sync > 0] = 1
    
    upchrip_data = data*upchrip_mask # Filter out none upchirp samples
    
    return upchrip_data
    
    

def extract_upchrip_indicies(sync):
    """
    Returns upchirp indicies
    * upchirp indicies: indicies where the upchirp starts
    """
    # Extract upchirp index
    chirp = np.zeros_like(sync)
    chirp[sync > 0] = 1
    chirp[sync < 0] = -1
    # Find zero crossing indecies
    zero_crossings = np.where(np.diff(np.sign(chirp)))[0]
    # Save all zero crossing that have an even index
    upchirp_indicies = zero_crossings[0::2] # Even index
    upchirp_indicies = upchirp_indicies[1:] # Remove first index since it is not a full chirp
    M = len(upchirp_indicies) # Number of chirps
    # Transform upchirp indecies to int
    upchirp_indicies = upchirp_indicies.astype(int)
    return upchirp_indicies

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




main()