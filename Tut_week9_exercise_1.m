% Exercise 1
clear; clc; close all;

I = im2double(rgb2gray(imread('plant4.jpg')));

% Gaussian smoothing
sigma = 1.2;
G = fspecial('gaussian', [5 5], sigma);
smoothed = imfilter(I, G, 'same');

% Sobel gradients
gx = [-1 0 1; -2 0 2; -1 0 1];
gy = [-1 -2 -1; 0 0 0; 1 2 1];
Gx = conv2(smoothed, gx, 'same');
Gy = conv2(smoothed, gy, 'same');
Gmag = sqrt(Gx.^2 + Gy.^2);

% Simple thresholding
T = 0.08 * max(Gmag(:));
edges_simple = Gmag > T;

% Built-in Canny for comparison
edges_builtin = edge(I, 'Canny', [0.08 0.2], sigma);

% Show the results
figure('Color','w');
subplot(1,3,1), imshow(I), title('Original (plant4)');
subplot(1,3,2), imshow(edges_simple), title('My Simple Canny');
subplot(1,3,3), imshow(edges_builtin), title('MATLAB edge(I, ''Canny'')');