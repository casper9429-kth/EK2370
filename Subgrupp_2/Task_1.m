clear all;
clc;

% Define the file path to your audio file
filePath = 'Velocity_Test_File.m4a';
[audioData, sampleRate] = audioread(filePath);
audioData_inv = audioData(:,1)*(-1);

N = length(audioData_inv);
T = 1/sampleRate;
Tp = 0.1;

%Amount of samples per sweeps, this is N from the slides
Sample_per_sweep=Tp/T;

%This is the number of sweeps
M = N/Sample_per_sweep;

%Creating a 2D array
First_array = zeros(round(M),Sample_per_sweep+4*Sample_per_sweep);

%Variable to keep track of each sample
k = 1;
for i = 1:M
    for j = 1:Sample_per_sweep
        First_array(i,j) = audioData_inv(k);
        k = k+1;
    end
    %Can we devide here to get the max for each row
    %FFt_array(i,:) = fft(First_array(i,:));
end
mean_first_array = mean(First_array(:,:));
final_first_array = First_array-mean_first_array;


for v = 1:M
    FFt_array(v,:) = fft(final_first_array(v,:));
end


f = linspace(0, sampleRate/2, Sample_per_sweep/2 + 1);
max_freq_value = zeros(1,round(M));
time_array = zeros(1,round(M));

t = 0;

for e = 1:M
    [~, index] = max(abs(FFt_array(e, :)));
    max_freq_value(e) = f(index);
    time_array(e) = t;
    t = t + Tp;
end
    
velocity_matrix = zeros(1,round(M));
carier_f = 2.43*10^9;
c = 3*10^8;

for h = 1:M
    velocity_matrix(h) = (max_freq_value(h)*c)/(2*carier_f);
end

for s = 1:M
    fft_norm(s,:) = abs( FFt_array(s,:)/max(abs(FFt_array(s,:))));
end

into_fft = fft_norm(:,:);

imagesc(velocity_matrix,time_array,into_fft);
xlim([0 150]);

%plot(time_array,10*log(velocity_matrix));
%ylim([0 100]);
