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

% Load the test file separately
testData = load(fullfile(dataFolder, testFile));
test_emg = testData.emg;
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
clearvars -except names trainData trainFiles test_emg test_stimulus

%% looping to extract emg and stimulus from training data

X_train = [];
Y_train = [];

for i = 1:numel(trainData)
% Process emg and stimulus as needed
% fprintf('Processing training file: %s\n', trainFiles(i).name);

    emg = trainData{i}.emg;
    stimulus = trainData{i}.stimulus;

    [segments, labels, T] = signalSegmenter(emg, stimulus, names);


    for i = 1:length(segments)
    [~, F1] = extractTimeDomainFeatures(segments{1,i});
    feat(i,:) = F1;
    end

    X_train = [X_train; feat];
    Y_train = [Y_train; labels'];
    
end
disp(T)
dat = table(X_train,Y_train)
%% Clearing variables
clearvars -except X_train Y_train test_emg test_stimulus names

%% Training the model
% Select Model Type
model_type = 'tree'; % Options: 'svm', 'knn', 'tree', 'nb', 'lda'

switch model_type
    case 'svm'
        model = fitcecoc(X_train, Y_train); % Multi-class SVM
    case 'knn'
        model = fitcknn(X_train, Y_train, 'NumNeighbors', 10); % KNN
    case 'tree'
        model = fitctree(X_train, Y_train); % Decision Tree
        view(model, 'Mode', 'graph');
    case 'nb'
        model = fitcnb(X_train, Y_train); % Naive Bayes
    case 'lda'
        model = fitcdiscr(X_train, Y_train); % Linear Discriminant Analysis (LDA)
    otherwise
        error('Invalid model type selected');
end

%Remarks for further improvement
% Reduce timing : use simpler model
% Improve accuracy : Use more advanced features

%% Testing the model

[segments2, labels2, T2] = signalSegmenter(test_emg, test_stimulus, names);

for i = 1:length(segments2)
[~, F2] = extractTimeDomainFeatures(segments2{1,i});
feat2(i,:) = F2;
end

X_test = feat2;
Y_test = labels2';

clearvars labels2 T2 F2 i feat2 model_type
%%

Y_pred = predict(model, X_test);
figure;
confusionchart(Y_test, Y_pred);

accuracy = sum(Y_pred == Y_test) / length(Y_test) * 100;
disp(['Test Accuracy: ', num2str(accuracy), '%']);

%% Plot Visualization (Ground Truth & Prediction in Subplots)

%Generating pred_stimulus
segment_lengths = cellfun(@length, segments2);
pred_stimulus = repelem(Y_pred, segment_lengths);

% Select EMG channel to visualize
channel_idx = 1;
emg_channel = test_emg(:, channel_idx);

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
         'Color', colors(stim_value + 1, :));
end
hold off;

% Create a shared colorbar
colormap(colors); % Apply the same colors used in the plot
cb = colorbar('Position', [0.92, 0.1, 0.02, 0.8]); % Adjust position for a shared colorbar
cb.Ticks = linspace(0, 1, length(names)); % Position ticks evenly
cb.TickLabels = names; % Assign movement labels
cb.Label.String = 'Movement Labels'; % Label for colorbar
