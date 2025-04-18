clc; clear; close all;

% Define the folder containing the .mat files
dataFolder = 'C:\Users\Dell\Documents\BE sem 2\Major Project\DB1_S1_E1'; 
filePattern = fullfile(dataFolder, '*.mat');
matFiles = dir(filePattern);

% Randomly select one file for testing
rng('shuffle'); % Randomness
randomIndex = randperm(numel(matFiles), 1);
testFile = matFiles(randomIndex).name;
trainFiles = matFiles;
trainFiles(randomIndex) = []; % Remove test file from training set

channel_idx = 7; %2,3,5, %7, %8, 9

% Load the test file separately
testData = load(fullfile(dataFolder, testFile));
test_emg = testData.emg;
test_emg = (test_emg(:,channel_idx))';
test_stimulus = testData.stimulus;

% Load all training files into a cell array
trainData = cell(1, numel(trainFiles));

for i = 1:numel(trainFiles)
    trainData{i} = load(fullfile(dataFolder, trainFiles(i).name));
end

% Display loaded file info
fprintf('Test file: %s\n', testFile);
fprintf('Training files loaded: %d\n', numel(trainFiles));

names = { 'Rest', 'Index Flexion', 'Index Extension', ...
           'Middle Flexion', 'Middle Extension', ...
           'Ring Flexion', 'Ring Extension', ...
           'Little Finger Flexion', 'Little Finger Extension', ...
           'Thumb Adduction', 'Thumb Abduction', ...
           'Thumb Flexion', 'Thumb Extension', };

%% Clearing variables
clearvars -except names trainData trainFiles test_emg test_stimulus channel_idx

%% looping to extract emg and stimulus from training data

X_train = [];
Y_train = [];
segments1 = [];
labels1 = [];

for i = 1:numel(trainData)
% Process emg and stimulus as needed
% fprintf('Processing training file: %s\n', trainFiles(i).name);

    emg = trainData{i}.emg;
    emg = (emg(:,channel_idx))';
    stimulus = trainData{i}.stimulus;

    [segments, labels, T] = signalSegmenter(emg, stimulus, names); 

    segments1 = [segments1 segments(1,:)];
    labels1 = [labels1 labels(1,:)];
end

[segments2, labels2, T2] = signalSegmenter(test_emg, test_stimulus, names);


%% Deep Learning

% Prepare Training Data
maxLength_combined = max([max(cellfun(@length, segments1)), max(cellfun(@length, segments2))]);
paddedSegments_train = cellfun(@(x) padarray(x, [0, maxLength_combined - length(x)], 0, 'post'), segments1, 'UniformOutput', false);


X_train = cell2mat(paddedSegments_train');
Y_train = categorical(labels1');

% Prepare Test Data
% Find the maximum length of all test segments
maxLength_test = max(cellfun(@length, segments2));
paddedSegments_test = cellfun(@(x) padarray(x, [0, maxLength_combined - length(x)], 0, 'post'), segments2, 'UniformOutput', false);

X_test = cell2mat(paddedSegments_test');
Y_test = categorical(labels2');

% Feature Scaling (Optional but Recommended)
X_train = normalize(X_train, 2); % Normalize along each row
X_test = normalize(X_test, 2);

%% Define Neural Network Structure
inputSize = size(X_train, 2);
numClasses = numel(unique(labels)); 

layers = [
    featureInputLayer(inputSize, 'Name', 'input')
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(numClasses, 'Name', 'fc3')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

%% Define Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {X_test, Y_test}, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%% Train the Network
net = trainNetwork(X_train, Y_train, layers, options);

%% Test the Model
Y_pred = classify(net, X_test);
accuracy = sum(Y_pred == Y_test) / numel(Y_test);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

%% Confusion Matrix
figure;

confusionchart(Y_test, Y_pred);
title('Confusion Matrix for Test Data');

%% Plot Visualization (Ground Truth & Prediction in Subplots)
Y_pred = double(Y_pred);
%Generating pred_stimulus
segment_lengths = cellfun(@length, segments2);
pred_stimulus = repelem(Y_pred, segment_lengths);

% Select EMG channel to visualize

emg_channel = test_emg;

% Time axis (assuming uniform sampling)
time = 1:length(emg_channel); % Adjust if you have actual time values

% Generate distinct colors for each movement
num_labels = length(names);
colors = distinguishable_colors(num_labels); % Generate well-separated colors
colors(1, :) = [0.8, 0.8, 0.8]; % Assign gray to 'Rest' (Label 0)

% Identify change points for ground truth and prediction
change_points_gt = [1; find(diff(test_stimulus) ~= 0) + 1; length(test_stimulus)];
change_points_pred = [1; find(diff(pred_stimulus) ~= 0) + 1; length(pred_stimulus)];

% Create figure with subplots
figure;

% Ground Truth Plot
subplot(2,1,1);
hold on;
xlabel('Samples');
ylabel('EMG Signal');
title(['Ground Truth - EMG Signal (Channel ', num2str(channel_idx), ')']);
grid on;

for i = 1:length(change_points_gt)-1
    idx_start = change_points_gt(i);
    idx_end = change_points_gt(i+1) - 1;
    stim_value = test_stimulus(idx_start);
    
    % Plot segment with corresponding color
    plot(time(idx_start:idx_end), emg_channel(idx_start:idx_end), ...
         'Color', colors(stim_value + 1, :));
end
hold off;

% Prediction Plot
subplot(2,1,2);
hold on;
xlabel('Samples');
ylabel('EMG Signal');
title(['Prediction - EMG Signal (Channel ', num2str(channel_idx), ')']);
grid on;

for i = 1:length(change_points_pred)-1
    idx_start = change_points_pred(i);
    idx_end = change_points_pred(i+1) - 1;
    stim_value = pred_stimulus(idx_start);
    
    % Plot segment with corresponding color
    plot(time(idx_start:idx_end), emg_channel(idx_start:idx_end), ...
         'Color', colors(stim_value + 0, :));
end
hold off;

% Create a shared colorbar
colormap(colors); % Apply the same colors used in the plot
cb = colorbar('Position', [0.79, 0.1, 0.02, 0.8]); % Adjust position for a shared colorbar
cb.Ticks = linspace(0, 1, length(names)); % Position ticks evenly
cb.TickLabels = names; % Assign movement labels
cb.Label.String = 'Movement Labels'; % Label for colorbar