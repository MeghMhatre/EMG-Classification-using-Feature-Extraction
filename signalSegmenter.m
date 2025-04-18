function [segments, labels, T] = signalSegmenter(signal, stimulus, names)

% Identify change points in stimulus
change_points = [1; find(diff(stimulus) ~= 0) + 1; length(stimulus)]; 

num_segments = length(change_points) - 1;
segments = cell(1, num_segments);
label_groups = cell(1, num_segments);

for i = 1:num_segments
    start_idx = change_points(i);
    end_idx = change_points(i+1) - 1;
    segments{i} = signal(start_idx:end_idx);
    label_groups{i} = stimulus(start_idx:end_idx);

    lab = label_groups{1,i};
    labels(i) = lab(1);
    movement_idx = labels(i);
    movement(i) = names(movement_idx+1);
end

% Create and display table
segment_names = strcat("S1 Seg ", string(1:num_segments)');
T = table(segment_names, segments', labels', movement', ...
    'VariableNames', {'Segment', 'Size', 'Labels', 'Names'});
%disp(T);

end