clear; clc;

inDir  = 'myHands/';   
outDir = 'edges/';     

if ~exist(outDir,'dir')
    mkdir(outDir);
end

imgs = dir(fullfile(inDir,'*.jpg'));

% Define the kernel of the three filters
% Prewitt
Px = [-1 0 1; -1 0 1; -1 0 1];
Py = [-1 -1 -1; 0 0 0; 1 1 1];

% Sobel
Sx = [-1 0 1; -2 0 2; -1 0 1];
Sy = [-1 -2 -1; 0 0 0; 1 2 1];

% Laplacian
Lap = [0 1 0; 1 -4 1; 0 1 0];

for k = 1:numel(imgs)
    name = imgs(k).name;
    I0 = imread(fullfile(inDir,name));
    if size(I0,3)==3, I = rgb2gray(I0); else, I = I0; end
    I = im2double(I);

    % Sobel：|Gx|+|Gy| 
    Gx_s = imfilter(I, Sx, 'replicate', 'conv');
    Gy_s = imfilter(I, Sy, 'replicate', 'conv');
    G_s  = abs(Gx_s) + abs(Gy_s);
    th_s = graythresh(mat2gray(G_s));            
    E_s  = imbinarize(mat2gray(G_s), th_s);

    % Prewitt：|Gx|+|Gy|
    Gx_p = imfilter(I, Px, 'replicate', 'conv');
    Gy_p = imfilter(I, Py, 'replicate', 'conv');
    G_p  = abs(Gx_p) + abs(Gy_p);
    th_p = graythresh(mat2gray(G_p));
    E_p  = imbinarize(mat2gray(G_p), th_p);

    % Laplacian
    Lresp = imfilter(I, Lap, 'replicate', 'conv');
    E_lap = edge(Lresp,'zerocross');

    % built-in function version
    E_sobel_builtin   = edge(I,'sobel'); 
    E_prewitt_builtin = edge(I,'prewitt');
    E_log_builtin     = edge(I,'log');

    % Save
    imwrite(E_s,               fullfile(outDir, ['SOBEL_manual_'   name]));
    imwrite(E_p,               fullfile(outDir, ['PREWITT_manual_' name]));
    imwrite(E_lap,             fullfile(outDir, ['LAPL_manual_'    name]));
    imwrite(E_sobel_builtin,   fullfile(outDir, ['SOBEL_edge_'     name]));
    imwrite(E_prewitt_builtin, fullfile(outDir, ['PREWITT_edge_'   name]));
    imwrite(E_log_builtin,     fullfile(outDir, ['LoG_edge_'       name]));

    % Visualising 6 results in one picture
    figure('Name',name);
    subplot(2,3,1); imshow(I);                 title(['Original - ' name]);
    subplot(2,3,2); imshow(E_s);               title('Sobel |Gx|+|Gy| (manual)');
    subplot(2,3,3); imshow(E_p);               title('Prewitt |Gx|+|Gy| (manual)');
    subplot(2,3,4); imshow(E_lap);             title('Laplacian Zero-Cross (manual)');
    subplot(2,3,5); imshow(E_sobel_builtin);   title('edge(''sobel'')');
    subplot(2,3,6); imshow(E_log_builtin);     title('edge(''log'')  (LoG)');
end
