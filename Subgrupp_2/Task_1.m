% Define the file path to your audio file
filePath = 'Velocity_Test_File.m4a';

% Read the audio file and get the audio data and sampling rate
[audioData, sampleRate] = audioread(filePath);

% Get the number of samples in the audio data
audioData_inv = audioData(:,1)*(-1);

N = length(audioData_inv);
T = 1/sampleRate;
Tp = 0.1;

%Amount of samples per sweeps
Sample_per_sweep=Tp/T;

%This is the number of sweeps
M = N/Sample_per_sweep;

%Creating a 2D array
First_array = zeros(round(M),Sample_per_sweep+4*length(N));

%Variable to keep track of each sample
k = 1;
for i = 1:M
    for j = 1:Sample_per_sweep
        First_array(i,j) = audioData_inv(k);
        k = k+1;
    end

    First_array(i,j)
    %hello

end



