% Define the file path to your audio file
filePath = 'Velocity_Test_File.m4a';

% Read the audio file and get the audio data and sampling rate
[audioData, sampleRate] = audioread(filePath);

% Get the number of samples in the audio data
N = length(audioData);

% Define the amount of zero-padding (4 times the original length)
zeroPaddingFactor = 4;
N_zeropad = N * zeroPaddingFactor;

% Perform zero-padding
audioData_zeropad = [audioData; zeros(N_zeropad - N, 2)];

% Perform FFT on the zero-padded audio data
Y = fft(audioData_zeropad);

% Compute the frequencies corresponding to the FFT result
frequencies = linspace(0, sampleRate, N_zeropad);

% Take the absolute value to get the magnitude spectrum
Y = abs(Y);

% Plot the frequency spectrum
figure;
plot(frequencies(1:N_zeropad/2), Y(1:N_zeropad/2));
title('Single-Sided Amplitude Spectrum');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
ylim([0 300]);
xlim([0.3e4 2e4]);


