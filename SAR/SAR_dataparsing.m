filePath = 'SAR_Test_File.m4a';

% Read the audio file and get the audio data and sampling rate
[audioData, sampleRate] = audioread(filePath);

% Get the number of samples in the audio data
audioData_inv = audioData(:,1)*(-1);