# Author: Casper Augustsson
# Transforms m4a file to data, sync and fs and saves it as a .mat file to be used in Matlab for further processing
# Enter path to m4a file in path variable. Must be a m4a file with two channels, one for data and one for sync
path = "SAR/SAR_measurement.m4a"

from pydub import AudioSegment
from pydub.utils import mediainfo
import numpy as np
import os
import scipy.io
import datetime
def main():
    
    # Get data, sync and fs
    data,sync,fs = read_data_sync_fs(path)
    
    # Get path to this file
    path_to_this_file = os.path.dirname(os.path.abspath(__file__))
    # Check time now as string
    time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Create a data_sync_fs folder if it does not exist
    if not os.path.exists(path_to_this_file + '/data_sync_fs'):
        os.makedirs(path_to_this_file + '/data_sync_fs')
    # Save data, sync and fs to .mat file with filename data_sync_fs_time_now.mat
    path_to_save = path_to_this_file + '/data_sync_fs/data_sync_fs_' + time_now + '.mat'
    
    scipy.io.savemat(path_to_save, mdict={'data': data,'sync':sync,'fs':fs})
    
    

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