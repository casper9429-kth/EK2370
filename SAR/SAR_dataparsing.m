clc;
clear all;
filePath = 'SAR_Test_File.m4a';

% Read the audio file and get the audio data and sampling rate
[audioData, sampleRate] = audioread(filePath);

% Get the number of samples in the audio data
%audioData_inv = audioData(:,1)*(-1);

%plot(audioData)
sync = audioData(:,2);
figure(1)
plot(sync)
hold on;
%%
%Extract position changes
syncsign = sync;
syncsign(syncsign < 0.008 & syncsign > -0.008) =0;
syncsign = sign(syncsign);
transitions = syncsign(2:end) - syncsign(1:end-1);
transitions = find(abs(transitions)>0);
figure(2);
plot(syncsign);

hold on
newpositions= find(diff(transitions)>40000) +1;
endpositions= find(diff(transitions)>40000);

plot(transitions(newpositions),0, "r*");
hold on;
plot(transitions(1), 0, "r*")
hold on;
plot(transitions(endpositions),0, "r*");
positionchanges = zeros(1, size(transitions(newpositions),1) +1)';
positionchanges(1)= transitions(1);
positionchanges(2:end) = transitions(newpositions);
positionchangesend = transitions(endpositions);
allpositionsignals = zeros(size(newpositions,1),0.5*sampleRate);
allpositionsyncs = zeros(size(newpositions,1),0.5*sampleRate);

for i = 1:size(positionchanges,1)
if i ~= size(positionchanges,1)
    allpositionsignals(i, :) = audioData(positionchanges(i)+0.5*sampleRate:positionchanges(i)+sampleRate-1,1);    
    allpositionsyncs(i,:) = sync(positionchanges(i)+0.5*sampleRate:positionchanges(i)+sampleRate-1);
end



end


%% 
