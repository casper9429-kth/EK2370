# This code intents to show how to use pydub to read m4a file
# And transform it to a numpy array
from pydub import AudioSegment
from pydub.utils import mediainfo
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.signal

def main():
    path = "data_cots/SAR_Test_File.m4a"
    
    # Get data, sync and fs
    data,sync,fs = read_data_sync_fs(path)


    # Constants
    c = 3e8;                # speed of light [m/s]
    Ts = 1/fs;              # sampling time [s]
    Trp = 0.5;              # Position update time [s] Trp>0.25 (lec 5,slide 32)
    Nrp=int(Trp*fs);        # Number of samples per position update
    Np = int(0.02*fs)       # Number of samples per upchirp
    Tp = 0.02;              # pulse width [s]
    fc = 2.43e9;            # center frequency [hz]
    Nsamples = int(Tp*fs);       # number of samples per puls
    BW = (2.495e9 - 2.408e9); # bandwidth of FMCW [Hz]
    padding_constant = 4;   # padding constant for the FFT

    # Find all indicies where a chirp starts
    # Find indicies where the chirp starts
    chirp_indicies,sync_clean = extract_chrip_indicies(sync)

    # Find most common difference between chirp indicies
    c_diff = chirp_indicies[1:] - chirp_indicies[:-1]
    chirp_dist = np.median(c_diff)    # Approximate distance between chirps

    # Find indicies where a new position starts in sync data
    pos_indicies = extract_pos_indicies(chirp_indicies)
        
    # # Plot sync and pos_indicies
    # pos_indicies_array = np.zeros_like(sync)
    # pos_indicies_array[pos_indicies] = np.max(sync)
    # plt.plot(sync)
    # plt.plot(pos_indicies_array)
    # plt.show()


    #### Create Matrix with all chirps
    # Cluster data in a matrix
    # i rows, where a row contains a position, 
    # j columns, where a column contains a sample. Have a total of Nrp columns

    # Remove the first Nrp/chirp_dist chirps for safety reasons
    chrips_to_remove = int(Nrp/chirp_dist)
    chirps_to_save = chrips_to_remove
    nr_pos = len(pos_indicies)
    
    # Create a matrix to house all the data
    data_matrix = np.zeros((nr_pos,Nrp))
    sync_matrix = np.zeros((nr_pos,Nrp))
    data_matrix_integrated = np.zeros((nr_pos,Np))
        
    # Iter through all rows and fill sync matrix and data matrix
    for i in range(nr_pos):
        
        # Get position index
        pos_index = pos_indicies[i]
        
        # Find index of pos_index in chirp_indicies, add chrips_to_remove to compensate for removed chirps and get the correct index
        index = np.where(chirp_indicies == pos_index)[0][0] + 2*chrips_to_remove
        # If the sync at index is negative, then the sync at index+1 is positive
        start_index = chirp_indicies[index] # This should correspond to the right amount of samples to remove
        if sync_clean[start_index] > 0.5:
            index = index+1
            start_index = chirp_indicies[index]
            
        # Create a matrix of all positive chirps added to each other
        # number of chirps
        # samples per chirp
        row_mat= np.zeros((chirps_to_save,Np))
        # Add every second chirp to row_mat starting at index
        for j in range(chirps_to_save):
            ind = index+j*2
            row_mat[j,:] = data[chirp_indicies[ind]:chirp_indicies[ind]+Np]
        # Sum all chirps together
        row_mat = np.mean(row_mat,axis=0) 
        data_matrix_integrated[i,:] = row_mat       

        
        # Fill in data_matrix with data and sync_matrix with data
        data_matrix[i,:] = data[start_index:start_index+Nrp]
        sync_matrix[i,:] = sync_clean[start_index:start_index+Nrp]
        # plt.plot(data_matrix[i,:Np])
        # plt.plot(data_matrix_integrated[i,:])
        # plt.plot(sync_matrix[i,:])
        # plt.show()
        
    
    # Peform das hillbert transform on data_matrix_integrated
    data_matrix_integrated_hillbert = das_hillbert_transform(data_matrix_integrated,1)

    # Plot first row
    plt.plot(data_matrix_integrated_hillbert[0,:])
    
    # Apply han windows to row
    data_matrix_integrated_hillbert = np.hanning(data_matrix_integrated_hillbert.shape[1])*data_matrix_integrated_hillbert

    # Add zero padding to data_matrix_integrated_hilbert
    data_matrix_integrated_hillbert_to_be_padded = np.zeros((data_matrix_integrated_hillbert.shape[0]+1012*2,data_matrix_integrated_hillbert.shape[1]))
    data_matrix_integrated_hillbert_to_be_padded[1012:-1012,:] = data_matrix_integrated_hillbert
    data_matrix_integrated_hillbert_padded = data_matrix_integrated_hillbert_to_be_padded
    
    
    # Apply fft along columns, fftshift along comums
    data_matrix_integrated_hillbert_fft = np.fft.fftshift(np.fft.fft(data_matrix_integrated_hillbert_padded,axis=0),axes=0)



def das_hillbert_transform(matrix,axis,rep_nan=1e-30):
    """
    Perform hilbert transform on matrix along axis and setting nan values to 1e-3
    """        
    # # Peform fft on matrix along axis
    # fft = np.fft.fft(matrix,axis=axis)

    # # Save positive frequencies of fft
    # fft = np.fft.fftshift(fft,axes=axis)
    
    # # Remove negative frequencies
    # if axis == 0:
    #     fft = fft[int(fft.shape[0]/2):,:]
    # else:
    #     fft = fft[:,int(fft.shape[1]/2):]        

    # # Replace NaN with rep_nan
    # fft[np.isnan(fft)] = rep_nan

    # return fft
    
    # Use scipy
    hilbert = scipy.signal.hilbert(matrix,axis=axis)
    # Replace nan values by rep_nan
    hilbert[np.isnan(hilbert)] = rep_nan
    return hilbert



    
def extract_pos_indicies(chirp_indicies):
    """
    Return indicies in sync where a new position starts
    """
    # Find all indicies where a new position starts
    # Calculate the difference between all values in chirp_indicies, 
    # remember it is still indicies
    c1 = chirp_indicies[1:] - chirp_indicies[:-1]

    # Set all indicies where is less than 50000 to 0, making them invalid
    c1[c1 < 50000] = 0
    c1[c1 >=50000] = 1   

    # All indicies where c1 corresponds to a index in chirp_indicies that is a pos index
    # Add elem in beginning of c1 with value 1 to compensate for the first removed index
    c1 = np.insert(c1,0,1)

    # Now use c1 to get indicies from chirp_indicies that corresponds to a pos index
    pos_indicies = chirp_indicies[c1 == 1]
    return pos_indicies

    
    
     
def extract_chrip_indicies(sync):
    """
    Returns chrip indicies
    * indicies: indicies where the chirp starts
    """
    #Extract upchirp index
    chirp = np.zeros_like(sync)
    threshold = 500
    chirp[sync > threshold] = 1
    chirp[sync < -threshold] = -1    
    
    # Find zero crossing indecies from positive to negative
    zero_crossings = np.where(np.diff(np.sign(chirp)))[0]
    # Save all zero crossing that have an even index
    #upchirp_indicies = zero_crossings[0::2] # Even index
    #upchirp_indicies = upchirp_indicies[1:] # Remove first index since it is not a full chirp
    #M = len(upchirp_indicies) # Number of chirps
    # Transform upchirp indecies to int
    zero_crossings = zero_crossings.astype(int)
    return zero_crossings,chirp




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



if __name__ == '__main__':
    main()