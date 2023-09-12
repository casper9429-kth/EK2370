filePath = 'Velocity_Test_File.m4a';

% Read the audio file and get the audio data and sampling rate
[audioData, sampleRate] = audioread(filePath);


audioData_inv = audioData(:,1)*(-1);


plot(audioData_inv);