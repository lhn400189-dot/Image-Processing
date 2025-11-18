%% Tutorial Week 7 - Frequency Domain Filters (week2_image)
clc; clear; close all;

% Paths
imgDir = 'C:\Users\LHN10\Desktop\COMP 478\tutorial_week7_Haonan_LI_40187232\week2_image';
outputDir = 'C:\Users\LHN10\Desktop\COMP 478\tutorial_week7_Haonan_LI_40187232\results';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Read all
imgFiles = [dir(fullfile(imgDir, '*.tif')); dir(fullfile(imgDir, '*.tiff'))];

% Loop through each image
for k = 1:length(imgFiles)

    % Step 1: Read safely
    [I, map] = imread(fullfile(imgDir, imgFiles(k).name));
    if ~isempty(map)
        I = ind2rgb(I, map);
    end
    if size(I,3) == 4
        I = I(:,:,1:3);
    end
    I = im2gray(I);
    I = im2double(I);

    % FFT Transform
    [M, N] = size(I);
    F = fft2(I);
    Fshift = fftshift(F);
    D0 = 50; n = 2;

    % Distance Matrix
    u = 0:(M-1); v = 0:(N-1);
    idx = find(u > M/2); u(idx) = u(idx) - M;
    idy = find(v > N/2); v(idy) = v(idy) - N;
    [V, U] = meshgrid(v, u);
    D = sqrt(U.^2 + V.^2);

    % Define Filters
    H_ideal_low = double(D <= D0);
    H_ideal_high = 1 - H_ideal_low;
    H_gaussian_low = exp(-(D.^2) / (2*(D0^2)));
    H_gaussian_high = 1 - H_gaussian_low;
    H_butter_low = 1 ./ (1 + (D./D0).^(2*n));
    H_butter_high = 1 - H_butter_low;

    filters = {H_ideal_low, H_gaussian_low, H_butter_low, ...
               H_ideal_high, H_gaussian_high, H_butter_high};
    names = {'Ideal Lowpass','Gaussian Lowpass','Butterworth Lowpass', ...
             'Ideal Highpass','Gaussian Highpass','Butterworth Highpass'};

    % Apply filters
    figure('Name', ['Filtering - ' imgFiles(k).name], 'NumberTitle', 'off');
    for i = 1:length(filters)
        G = Fshift .* filters{i};
        G = ifftshift(G);
        filtered = real(ifft2(G));
        subplot(2,3,i);
        imshow(mat2gray(filtered));
        title(names{i});

        prefix = erase(imgFiles(k).name, {'.tif', '.tiff'});
        outName = sprintf('%s_%s.png', prefix, strrep(names{i}, ' ', '_'));
        imwrite(mat2gray(filtered), fullfile(outputDir, outName));
    end

    comparisonName = sprintf('%s_all_filters.png', erase(imgFiles(k).name, {'.tif', '.tiff'}));
    saveas(gcf, fullfile(outputDir, comparisonName));

    pause(0.3);
    close all;
end


%% Tutorial Week 7 - Frequency Domain Filters (hand_image)
clc; clear; close all;

% Paths
imgDir = 'C:\Users\LHN10\Desktop\COMP 478\tutorial_week7_Haonan_LI_40187232\myHands';
outputDir = 'C:\Users\LHN10\Desktop\COMP 478\tutorial_week7_Haonan_LI_40187232\results_myHands';

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Read all
imgFiles = [dir(fullfile(imgDir, '*.jpg')); dir(fullfile(imgDir, '*.jpeg'))];

% Loop through each image
for k = 1:length(imgFiles)

    % Step 1: Read safely
    [I, map] = imread(fullfile(imgDir, imgFiles(k).name));
    if ~isempty(map)
        I = ind2rgb(I, map);
    end
    if size(I,3) == 4
        I = I(:,:,1:3);
    end
    I = im2gray(I);
    I = im2double(I);

    % FFT Transform
    [M, N] = size(I);
    F = fft2(I);
    Fshift = fftshift(F);
    D0 = 50; n = 2;

    % Distance Matrix
    u = 0:(M-1); v = 0:(N-1);
    idx = find(u > M/2); u(idx) = u(idx) - M;
    idy = find(v > N/2); v(idy) = v(idy) - N;
    [V, U] = meshgrid(v, u);
    D = sqrt(U.^2 + V.^2);

    % Define Filters
    H_ideal_low = double(D <= D0);
    H_ideal_high = 1 - H_ideal_low;
    H_gaussian_low = exp(-(D.^2) / (2*(D0^2)));
    H_gaussian_high = 1 - H_gaussian_low;
    H_butter_low = 1 ./ (1 + (D./D0).^(2*n));
    H_butter_high = 1 - H_butter_low;

    filters = {H_ideal_low, H_gaussian_low, H_butter_low, ...
               H_ideal_high, H_gaussian_high, H_butter_high};
    names = {'Ideal Lowpass','Gaussian Lowpass','Butterworth Lowpass', ...
             'Ideal Highpass','Gaussian Highpass','Butterworth Highpass'};

    % Apply filters
    figure('Name', ['Filtering - ' imgFiles(k).name], 'NumberTitle', 'off');
    for i = 1:length(filters)
        G = Fshift .* filters{i};
        G = ifftshift(G);
        filtered = real(ifft2(G));
        subplot(2,3,i);
        imshow(mat2gray(filtered));
        title(names{i});

        prefix = erase(imgFiles(k).name, {'.tif', '.tiff'});
        outName = sprintf('%s_%s.png', prefix, strrep(names{i}, ' ', '_'));
        imwrite(mat2gray(filtered), fullfile(outputDir, outName));
    end

    comparisonName = sprintf('%s_all_filters.png', erase(imgFiles(k).name, {'.tif', '.tiff'}));
    saveas(gcf, fullfile(outputDir, comparisonName));

    pause(0.3);
    close all;
end


