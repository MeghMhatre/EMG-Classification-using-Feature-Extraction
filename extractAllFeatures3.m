function [features, F] = extractAllFeatures3(signal, Fs)
    % Extracts combined time-domain and advanced features:
    % MAV, RMS, WL, SSC, IAV, DWT, VAR, ENT + WST with PCA.

    % --- Basic Time-Domain Features ---
    features.MAV = meanAbsoluteValue(signal);
    features.RMS = rootMeanSquare(signal);
    features.WL = waveformLength(signal);
    features.SSC = slopeSignChanges(signal);

    % --- Advanced Features ---
    features.IAV = integratedAbsoluteValue(signal);
    features.DWT = dwtFeature(signal, 'db7');
    features.VAR = varianceFeature(signal);
    features.ENT = entropyFeature(signal);

    % --- Set Number of PCA Components for WST ---
    pcaComponents = 2;  % Desired number of PCA components
    sf = waveletScattering('SignalLength', numel(signal), 'SamplingFrequency', Fs);
    scat_features = featureMatrix(sf, signal);
    scat_features = permute(scat_features, [2 3 1]);
    FSWT = reshape(scat_features, 1, []); % Flatten into a row vector

    % --- Apply PCA for WST Features ---
    if length(FSWT) >= pcaComponents
        [~, score, ~] = pca(FSWT');  % PCA on transposed vector
        FSWT = score(1:pcaComponents)';  % Take first 'pcaComponents' as reduced features
    else
        % Pad with zeros if feature length is less than required PCA components
        FSWT = [FSWT, zeros(1, pcaComponents - length(FSWT))];
    end

    % --- Store WST Features ---
    for i = 1:pcaComponents
        features.(['WST_PCA' num2str(i)]) = FSWT(i);
    end

    % --- Create Table with Feature Names and Values ---
    featureNames = fieldnames(features);
    featureValues = struct2cell(features)';
    F = cell2table(featureValues, 'VariableNames', featureNames);
end

% --- Mean Absolute Value (MAV) ---
function mav = meanAbsoluteValue(signal)
    mav = mean(abs(signal));
end

% --- Root Mean Square (RMS) ---
function rms_val = rootMeanSquare(signal)
    rms_val = sqrt(mean(signal.^2));
end

% --- Waveform Length (WL) ---
function wl = waveformLength(signal)
    wl = sum(abs(diff(signal)));
end

% --- Slope Sign Changes (SSC) ---
function ssc = slopeSignChanges(signal)
    N = length(signal);
    ssc = 0;
    for i = 2:N-1
        if (signal(i) > signal(i-1) && signal(i) > signal(i+1)) || ...
           (signal(i) < signal(i-1) && signal(i) < signal(i+1))
            ssc = ssc + 1;
        end
    end
end

% --- Integrated Absolute Value (IAV) ---
function iav = integratedAbsoluteValue(signal)
    iav = sum(abs(signal));
end

% --- Discrete Wavelet Transform (DWT) Feature ---
function dwt_val = dwtFeature(signal, wavelet)
    [C, ~] = wavedec(signal, 4, wavelet); % 4-level DWT
    dwt_val = sum(abs(C)); % Sum of absolute wavelet coefficients
end

% --- Variance (VAR) ---
function var_val = varianceFeature(signal)
    var_val = var(signal);
end

% --- Entropy (ENT) ---
function ent_val = entropyFeature(signal)
    p = histcounts(signal, 100) / length(signal);
    p(p == 0) = [];
    ent_val = -sum(p .* log2(p));
end
