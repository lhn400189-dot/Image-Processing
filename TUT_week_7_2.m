%% Tutorial Week 7 - (2) Laplacian Filter: Spatial vs Frequency Domain
clc; clear; close all;

% Paths
imgDir = 'C:\Users\LHN10\Desktop\COMP 478\tutorial_week7_Haonan_LI_40187232\week2_image';
outputDir = 'C:\Users\LHN10\Desktop\COMP 478\tutorial_week7_Haonan_LI_40187232\results_week2_Laplacian';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Load all Week2 images
imgFiles = [dir(fullfile(imgDir, '*.jpg')); dir(fullfile(imgDir, '*.png')); ...
            dir(fullfile(imgDir, '*.tif')); dir(fullfile(imgDir, '*.tiff')); dir(fullfile(imgDir, '*.bmp'))];

for k = 1:length(imgFiles)
    fprintf('\nProcessing Laplacian for: %s\n', imgFiles(k).name);

    % Read image
    [I, map] = imread(fullfile(imgDir, imgFiles(k).name));
    if ~isempty(map)
        I = ind2rgb(I, map);
    end
    if size(I,3) == 4
        I = I(:,:,1:3);
    end
    I = im2gray(I);
    I = im2double(I);
    [M, N] = size(I);

    %  Spatial Laplacian
    h = fspecial('laplacian', 0);
    spatialLap = imfilter(I, h, 'replicate');

    % Frequency Laplacian
    F = fft2(I);
    Fshift = fftshift(F);
    u = 0:(M-1); v = 0:(N-1);
    idx = find(u > M/2); u(idx) = u(idx) - M;
    idy = find(v > N/2); v(idy) = v(idy) - N;
    [V, U] = meshgrid(v, u);
    D = sqrt(U.^2 + V.^2);
    H = -4 * (pi^2) * (D.^2);

    G = H .* Fshift;
    G = ifftshift(G);
    freqLap = real(ifft2(G));

    % Display Comparison
    figure('Name', ['Laplacian Comparison - ' imgFiles(k).name], 'NumberTitle', 'off');
    subplot(1,3,1); imshow(I, []); title('Original Image');
    subplot(1,3,2); imshow(mat2gray(spatialLap)); title('Spatial Laplacian');
    subplot(1,3,3); imshow(mat2gray(freqLap)); title('Frequency Laplacian');

    % Plot Frequency Filter Surface
    figure('Name', ['Laplacian Filter - ' imgFiles(k).name], 'NumberTitle', 'off');
    mesh(H(floor(M/2)-50:floor(M/2)+50, floor(N/2)-50:floor(N/2)+50));
    title('Laplacian Filter in Frequency Domain');
    xlabel('u'); ylabel('v'); zlabel('H(u,v)');

    % Save results
    prefix = erase(imgFiles(k).name, {'.jpg','.png','.tif','.tiff','.bmp'});
    imwrite(mat2gray(spatialLap), fullfile(outputDir, [prefix '_Spatial_Laplacian.png']));
    imwrite(mat2gray(freqLap), fullfile(outputDir, [prefix '_Frequency_Laplacian.png']));
    saveas(gcf, fullfile(outputDir, [prefix '_Laplacian_Filter_Surface.png']));

    pause(0.2);
    close all;
end


%% Tutorial Week 7 - (2) Laplacian Filter for myHands
clc; clear; close all;

% Paths
imgDir = 'C:\Users\LHN10\Desktop\COMP 478\tutorial_week7_Haonan_LI_40187232\myHands';
outputDir = 'C:\Users\LHN10\Desktop\COMP 478\tutorial_week7_Haonan_LI_40187232\results_myHands_Laplacian';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Read all hand images
imgFiles = [dir(fullfile(imgDir, '*.jpg')); dir(fullfile(imgDir, '*.jpeg')); ...
            dir(fullfile(imgDir, '*.png')); dir(fullfile(imgDir, '*.tif')); dir(fullfile(imgDir, '*.tiff'))];

for k = 1:length(imgFiles)
    fprintf('\nProcessing Laplacian for: %s\n', imgFiles(k).name);

    % Read safely
    [I, map] = imread(fullfile(imgDir, imgFiles(k).name));
    if ~isempty(map)
        I = ind2rgb(I, map);
    end
    if size(I,3) == 4
        I = I(:,:,1:3);
    end
    I = im2gray(I);
    I = im2double(I);
    [M, N] = size(I);

    % Spatial Laplacian
    h = fspecial('laplacian', 0);
    spatialLap = imfilter(I, h, 'replicate');

    % Frequency Laplacian
    F = fft2(I);
    Fshift = fftshift(F);
    u = 0:(M-1); v = 0:(N-1);
    idx = find(u > M/2); u(idx) = u(idx) - M;
    idy = find(v > N/2); v(idy) = v(idy) - N;
    [V, U] = meshgrid(v, u);
    D = sqrt(U.^2 + V.^2);
    H = -4 * (pi^2) * (D.^2);
    G = H .* Fshift;
    G = ifftshift(G);
    freqLap = real(ifft2(G));

    % Display Comparison
    figure('Name', ['Laplacian Comparison - ' imgFiles(k).name], 'NumberTitle', 'off');
    subplot(1,3,1); imshow(I, []); title('Original Image');
    subplot(1,3,2); imshow(mat2gray(spatialLap)); title('Spatial Laplacian');
    subplot(1,3,3); imshow(mat2gray(freqLap)); title('Frequency Laplacian');

    % Plot Frequency Filter Surface
    figure('Name', ['Laplacian Filter - ' imgFiles(k).name], 'NumberTitle', 'off');
    mesh(H(floor(M/2)-50:floor(M/2)+50, floor(N/2)-50:floor(N/2)+50));
    title('Laplacian Filter in Frequency Domain');
    xlabel('u'); ylabel('v'); zlabel('H(u,v)');
    ax = gca; ax.Toolbar.Visible = 'off';  

    % Save results
    prefix = erase(imgFiles(k).name, {'.jpg','.jpeg','.png','.tif','.tiff'});
    imwrite(mat2gray(spatialLap), fullfile(outputDir, [prefix '_Spatial_Laplacian.png']));
    imwrite(mat2gray(freqLap), fullfile(outputDir, [prefix '_Frequency_Laplacian.png']));
    saveas(gcf, fullfile(outputDir, [prefix '_Laplacian_Filter_Surface.png']));


    pause(0.2);
    close all;
end

