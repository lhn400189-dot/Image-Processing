% Setting the image folder path
imgFolder = 'C:\Users\LHN10\Desktop\COMP 478\tutorial_week8_Haonan_LI_40187232\myHands';
imgFiles = dir(fullfile(imgFolder, '*.jpg'));

% Setting the filter parameters
LEN = 20;
THETA = 10;
K = 0.01;    
PSF = fspecial('motion', LEN, THETA);

%  Process each hand image in turn
for i = 1:length(imgFiles)
    fileName = fullfile(imgFolder, imgFiles(i).name);
    I = im2double(imread(fileName));
    if size(I,3) == 3
        I = rgb2gray(I);
    end

    % Add motion blur
    blurred = imfilter(I, PSF, 'conv', 'circular');

    % Adding Gaussian Noise
    noisy = imnoise(blurred, 'gaussian', 0, 0.001);

    % Pseudo Inverse Filter
    H = psf2otf(PSF, size(I));
    G = fft2(noisy);
    F_hat = G ./ H;
    F_hat(abs(H) < 0.9) = G(abs(H) < 0.9) ./ 0.9;
    restored_pseudo = real(ifft2(F_hat));

    % Wiener Filter
    restored_wiener = deconvwnr(noisy, PSF, K);

    % Show results
    figure('Name', imgFiles(i).name, 'NumberTitle', 'off');
    subplot(2,2,1), imshow(I), title('Original');
    subplot(2,2,2), imshow(noisy), title('Blurred + Noise');
    subplot(2,2,3), imshow(restored_pseudo, []), title('Pseudo Inverse');
    subplot(2,2,4), imshow(restored_wiener, []), title('Wiener Filter');

    saveas(gcf, fullfile(imgFolder, ['result_' imgFiles(i).name '.png']));
end