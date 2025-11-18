% Set parameters
sigma = 1.1;
T_ratio = 0.09;
T_low  = 0.06;
T_high = 0.18;

in_dir  = 'C:\Users\LHN10\Desktop\COMP 478\tutorial_week9_Haonan_LI_40187232\myHands';
out_dir = fullfile(in_dir, 'results');
if ~exist(out_dir,'dir'), mkdir(out_dir); end

files = [dir(fullfile(in_dir,'*.jpg')); dir(fullfile(in_dir,'*.png')); dir(fullfile(in_dir,'*.jpeg'))];
fprintf('Found %d images in "%s"\n', numel(files), in_dir);

for k = 1:numel(files)
    fname = files(k).name;
    I0 = im2double(imread(fullfile(in_dir, fname)));
    if size(I0,3)==3, I = rgb2gray(I0); else, I = I0; end
    
    % Increase contrast
    I = imadjust(I);
    
    % Sobel gradients
    G = fspecial('gaussian', [5 5], sigma);
    smooth = imfilter(I, G, 'same');
    gx = [-1 0 1; -2 0 2; -1 0 1];
    gy = [-1 -2 -1; 0 0 0; 1 2 1];
    Gx = conv2(smooth, gx, 'same');
    Gy = conv2(smooth, gy, 'same');
    Gmag = sqrt(Gx.^2 + Gy.^2);
    T = T_ratio * max(Gmag(:));
    edges_simple = Gmag > T;
    
    % Canny
    edges_builtin = edge(I, 'Canny', [T_low T_high], sigma);
    
    % Display and Save
    f = figure('Color','w','Position',[100 100 1200 400]);
    subplot(1,3,1), imshow(I0,[]), title(sprintf('Original - %s', fname),'Interpreter','none');
    subplot(1,3,2), imshow(edges_simple,[]), title('My Simple Canny');
    subplot(1,3,3), imshow(edges_builtin,[]), title('MATLAB edge(I, ''Canny'')');
    saveas(f, fullfile(out_dir, sprintf('result_%02d_%s.png',k,fname)));
    close(f);

    fprintf('[%02d/%02d] %-20s  MyEdges=%6d | Built-in=%6d\n',...
        k, numel(files), fname, nnz(edges_simple), nnz(edges_builtin));
end