clear; clc;

% Setting up the dataset folder
input_folder = 'myHands/';
output_folder = 'results/';

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Get all image files
img_files = dir(fullfile(input_folder, '*.jpg'));

% Define filter size
N = 3;

for i = 1:length(img_files)
    img_name = img_files(i).name;
    img_path = fullfile(input_folder, img_name);
    img = imread(img_path);

    % Conversion to grey scale
    if size(img,3) == 3
        gray = rgb2gray(img);
    else
        gray = img;
    end
    
    % 1. Negative filter
    negative = 255 - gray;

    % 2. Median filter
    medianFiltered = medfilt2(gray, [N N]);

    % 3. Intensity transformation
    gamma = 0.5;
    gammaCorrected = im2uint8(mat2gray(double(gray) .^ gamma));

    % Contrast stretching
    contrastStretched = imadjust(gray, stretchlim(gray, [0.01 0.99]), []);

    % % Save results
    imwrite(negative, fullfile(output_folder, ['NEG_' img_name]));
    imwrite(medianFiltered, fullfile(output_folder, ['MED_' img_name]));
    imwrite(gammaCorrected, fullfile(output_folder, ['GAMMA_' img_name]));
    imwrite(contrastStretched, fullfile(output_folder, ['CONTRAST_' img_name]));

    % Show results
    figure
    subplot(2,3,1); imshow(gray); title(['Original - ' img_name]);
    subplot(2,3,2); imshow(negative); title('Negative');
    subplot(2,3,3); imshow(medianFiltered); title(['Median ' num2str(N) 'x' num2str(N)]);
    subplot(2,3,4); imshow(gammaCorrected); title(['Gamma \gamma=' num2str(gamma)]);
    subplot(2,3,5); imshow(contrastStretched); title('Contrast Stretching');
end