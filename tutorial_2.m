% 1. Read the images and display them
grayFile = 'lena512.bmp';
colorFile = 'lena512color.tiff';

grayImg = imread(grayFile);
colorImg = imread(colorFile);

figure, imshow(grayImg), title('GrayImg');
figure, imshow(colorImg), title('colorImg');

% 2. (a) Convert the color image to grayscale
Icolor = imread("lena512color.tiff");
Icolor_gray = rgb2gray(Icolor);
figure, imshow(Icolor_gray), title('Colour to greyscale Img');

% (b) Convert the grayscale image into a binary image
Igray = imread('lena512.bmp');
threshold = 128;
Ibin  = Igray > threshold;
figure, imshow(Ibin), title('Greyscale to Binary IMG');

% (c) Negative the grayscale image
Igray = imread('lena512.bmp');
Ineg = imcomplement(Igray);  
imshow(Ineg), title('Negative of Grayscale IMG');

% (d) d. Write a function that return 3 channels (R, G, and B) of the color image as 3 new images
Icolor = imread("lena512color.tiff");
R = Icolor(:,:,1);  
G = Icolor(:,:,2);   
B = Icolor(:,:,3);  
% Display the individual color channels
figure, imshow(R), title('Red Channel');
figure, imshow(G), title('Green Channel');
figure, imshow(B), title('Blue Channel');

% (e) Extract a subimage size M x N centering at (x, y)
Igray = imread('lena512.bmp');
[r, c] = size(Igray);
M = 100;   % height of subimage
N = 100;   % width of subimage
x = 200;   % center column
y = 200;   % center row
rows = (y - floor(M/2)) : (y + floor(M/2) - 1);
cols = (x - floor(N/2)) : (x + floor(N/2) - 1);
subImg = Igray(rows, cols);
% Display result
figure, imshow(subImg), title('Subimage extracted by matrix indexing');

% (f) Write two functions to flip an image vertically and horizontally
function J = flipVertical(I)
    J = I(end:-1:1, :, :);
end

function J = flipHorizontal(I)
    J = I(:, end:-1:1, :);
end

Igray = imread('lena512.bmp');

Iv = flipVertical(Igray);
Ih = flipHorizontal(Igray);

figure;
subplot(1,3,1), imshow(Igray), title('Original');
subplot(1,3,2), imshow(Iv),    title('Vertical Flip');
subplot(1,3,3), imshow(Ih),    title('Horizontal Flip');

% (g) Write a function to rotate an image 90 degree left or right
function J = rotate90(I, direction)
    switch lower(direction)
        case {'left','ccw'}  
            J = permute(I, [2 1 3]); 
            J = J(end:-1:1, :, :);     
        case {'right','cw'}   
            J = permute(I, [2 1 3]);
            J = J(:, end:-1:1, :);     
        otherwise
            error('Direction must be ''left'' or ''right''');
    end
end
Igray = imread('lena512.bmp');

Ileft  = rotate90(Igray, 'left');
Iright = rotate90(Igray, 'right');

figure;
subplot(1,3,1), imshow(Igray),  title('Original');
subplot(1,3,2), imshow(Ileft),  title('Rotate Left 90°');
subplot(1,3,3), imshow(Iright), title('Rotate Right 90°');

% 3. Display all the result images together with the original images in the same figure
figure('Name','All Results');

subplot(3,5,1), imshow(grayImg),       title('Original Gray');
subplot(3,5,2), imshow(colorImg),      title('Original Color');
subplot(3,5,3), imshow(Icolor_gray),   title('Color→Gray');
subplot(3,5,4), imshow(Ibin),          title('Gray→Binary');
subplot(3,5,5), imshow(Ineg),          title('Negative');

subplot(3,5,6), imshow(R),             title('Red Channel');
subplot(3,5,7), imshow(G),             title('Green Channel');
subplot(3,5,8), imshow(B),             title('Blue Channel');
subplot(3,5,9), imshow(subImg),        title('Subimage');
subplot(3,5,10), imshow(Iv),           title('Vertical Flip');

subplot(3,5,11), imshow(Ih),           title('Horizontal Flip');
subplot(3,5,12), imshow(Ileft),        title('Rotate Left 90°');
subplot(3,5,13), imshow(Iright),       title('Rotate Right 90°');

% 4. Save all the result images into a subfolder

% Create folder 'results' if it does not exist
outDir = fullfile(pwd, 'results');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% Save all results
imwrite(grayImg,      fullfile(outDir,'result_gray.png'));
imwrite(colorImg,     fullfile(outDir,'result_color.png'));
imwrite(Icolor_gray,  fullfile(outDir,'result_color2gray.png'));
imwrite(Ibin,         fullfile(outDir,'result_binary.png'));
imwrite(Ineg,         fullfile(outDir,'result_negative.png'));

imwrite(R,            fullfile(outDir,'result_R.png'));
imwrite(G,            fullfile(outDir,'result_G.png'));
imwrite(B,            fullfile(outDir,'result_B.png'));
imwrite(subImg,       fullfile(outDir,'result_subimage.png'));

imwrite(Iv,           fullfile(outDir,'result_vertical_flip.png'));
imwrite(Ih,           fullfile(outDir,'result_horizontal_flip.png'));
imwrite(Ileft,        fullfile(outDir,'result_rotate_left90.png'));
imwrite(Iright,       fullfile(outDir,'result_rotate_right90.png'));

fprintf('All result images have been saved in: %s\n', outDir);

% 5.Display the information of the result images

resultFiles = { ...
    'result_gray.png', ...
    'result_color.png', ...
    'result_color2gray.png', ...
    'result_binary.png', ...
    'result_negative.png', ...
    'result_R.png', ...
    'result_G.png', ...
    'result_B.png', ...
    'result_subimage.png', ...
    'result_vertical_flip.png', ...
    'result_horizontal_flip.png', ...
    'result_rotate_left90.png', ...
    'result_rotate_right90.png'};

fprintf('\nImage Information:\n');
for k = 1:numel(resultFiles)
    filePath = fullfile(outDir, resultFiles{k});
    info = imfinfo(filePath);
    fprintf('File: %s\n', resultFiles{k});
    fprintf('  Format: %s\n', info.Format);
    fprintf('  Size: %d x %d\n', info.Width, info.Height);
    fprintf('  BitDepth: %d\n\n', info.BitDepth);
end
