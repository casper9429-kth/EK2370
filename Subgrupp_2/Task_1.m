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
First_array = zeros(round(M)-1,Sample_per_sweep);

%Variable to keep track of each sample
k = 1;
velocities = linspace(0,sampleRate/2, 2*Sample_per_sweep);
velocities = velocities * (3*10^8)/(2 * 2.43 * 10^9);

timearray = linspace(0,Tp*(round(M)-1),round(M)-1);


audioData_inv = reshape(audioData_inv(1:Sample_per_sweep*(round(M)-1)), [Sample_per_sweep, round(M)-1]);
First_array(:, 1:Sample_per_sweep) = audioData_inv';

%clutter rej
First_array(:,1:Sample_per_sweep) = First_array(:,1:Sample_per_sweep) - mean(First_array(:,1:Sample_per_sweep),"all");
figure(2)
fftfirst = 10*log10(abs(fft(First_array,5*Sample_per_sweep,2)));
fftfirst = fftfirst(:,1:Sample_per_sweep*2);
%Norm1
%maxall = max(fftfirst, [], 'all');
%fftfirst = fftfirst - maxall;
%Norm2
maxrows = max(fftfirst,[], 2);
fftfirst = fftfirst - maxrows;
imagesc(velocities, timearray, fftfirst,[-20 0])
xlim([0 30])










