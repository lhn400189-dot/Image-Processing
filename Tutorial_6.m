inDir  = "C:\Users\LHN10\Desktop\COMP 478\tutorial_week6\myHands";
outDir = "C:\Users\LHN10\Desktop\COMP 478\tutorial_week6";  

% Reproduce each of the filters (highpass and lowpass) and plot them (6 figures)
% Try to read the first image to get MxN 
files = [dir(fullfile(inDir,'*.jpg')); dir(fullfile(inDir,'*.png')); dir(fullfile(inDir,'*.jpeg'))];
if ~isempty(files)
    I0 = imread(fullfile(inDir, files(1).name));
    if size(I0,3)==3, I = rgb2gray(I0); else, I = I0; end
    I = im2double(I);
    [M,N] = size(I);
else
    warning('No images found. Using a default size of 512x512 for filter design.');
    M = 512; N = 512;
end

% horizontal frequency axis, V: vertical frequency axis
[U,V] = meshgrid(-N/2:N/2-1, -M/2:M/2-1);
D = sqrt(U.^2 + V.^2);   

D0 = 50;   
n  = 2;  

% Low-pass filters
H_ideal_LP   = double(D <= D0);                    
H_butter_LP  = 1 ./ (1 + (D./D0).^(2*n));             
H_gauss_LP   = exp(-(D.^2) / (2*D0^2));                

% High-pass filters 
H_ideal_HP   = 1 - H_ideal_LP;                   
H_butter_HP  = 1 - H_butter_LP;                      
H_gauss_HP   = 1 - H_gauss_LP;                       

% Helper to plot and save a mesh 
function save_mesh(U,V,H,titleText,saveName,outDir)
    fig = figure('Color','w','Name',titleText,'NumberTitle','off');
    mesh(U, V, H);  
    xlabel('u (freq)'); ylabel('v (freq)'); zlabel('H(u,v)');
    title(titleText);
    axis tight; view(45,35); grid on;
    drawnow;
    exportgraphics(fig, fullfile(outDir, saveName), 'Resolution', 200);
end

% Plot & save all six meshes
save_mesh(U,V,H_ideal_LP,  sprintf('Ideal LPF (D0=%d)',D0),     'mesh_Ideal_LPF.png',  outDir);
save_mesh(U,V,H_butter_LP, sprintf('Butterworth LPF (D0=%d,n=%d)',D0,n), 'mesh_Butterworth_LPF.png', outDir);
save_mesh(U,V,H_gauss_LP,  sprintf('Gaussian LPF (D0=%d)',D0),  'mesh_Gaussian_LPF.png', outDir);

save_mesh(U,V,H_ideal_HP,  sprintf('Ideal HPF (D0=%d)',D0),     'mesh_Ideal_HPF.png',  outDir);
save_mesh(U,V,H_butter_HP, sprintf('Butterworth HPF (D0=%d,n=%d)',D0,n), 'mesh_Butterworth_HPF.png', outDir);
save_mesh(U,V,H_gauss_HP,  sprintf('Gaussian HPF (D0=%d)',D0),  'mesh_Gaussian_HPF.png', outDir);

% Apply each of the filter you designed and plot your results.
% Read one hand image
I0 = imread(fullfile(inDir, 'RBC.jpg'));   
if size(I0,3)==3
    I = rgb2gray(I0);
else
    I = I0;
end
I = im2double(I);
[M,N] = size(I);

% Compute Fourier transform and shift DC component to center
F = fft2(I);
Fshift = fftshift(F);

% Apply the six filters in the frequency domain
G_ideal_LP  = Fshift .* H_ideal_LP;
G_butter_LP = Fshift .* H_butter_LP;
G_gauss_LP  = Fshift .* H_gauss_LP;

G_ideal_HP  = Fshift .* H_ideal_HP;
G_butter_HP = Fshift .* H_butter_HP;
G_gauss_HP  = Fshift .* H_gauss_HP;

% Inverse transform back to spatial domain
I_ideal_LP  = real(ifft2(ifftshift(G_ideal_LP)));
I_butter_LP = real(ifft2(ifftshift(G_butter_LP)));
I_gauss_LP  = real(ifft2(ifftshift(G_gauss_LP)));

I_ideal_HP  = real(ifft2(ifftshift(G_ideal_HP)));
I_butter_HP = real(ifft2(ifftshift(G_butter_HP)));
I_gauss_HP  = real(ifft2(ifftshift(G_gauss_HP)));

% Display results
figure('Name','Filter Applications','Color','w');
subplot(3,3,1); imshow(I,[]); title(['Original - ' 'RBC.jpg']);
subplot(3,3,2); imshow(mat2gray(I_ideal_LP));  title('Ideal LPF');
subplot(3,3,3); imshow(mat2gray(I_ideal_HP));  title('Ideal HPF');
subplot(3,3,4); imshow(mat2gray(I_butter_LP)); title('Butterworth LPF');
subplot(3,3,5); imshow(mat2gray(I_butter_HP)); title('Butterworth HPF');
subplot(3,3,6); imshow(mat2gray(I_gauss_LP));  title('Gaussian LPF');
subplot(3,3,7); imshow(mat2gray(I_gauss_HP));  title('Gaussian HPF');

% Save filtered images
imwrite(mat2gray(I_ideal_LP),  fullfile(outDir, 'RBC_IdealLPF.png'));
imwrite(mat2gray(I_ideal_HP),  fullfile(outDir, 'RBC_IdealHPF.png'));
imwrite(mat2gray(I_butter_LP), fullfile(outDir, 'RBC_ButterLPF.png'));
imwrite(mat2gray(I_butter_HP), fullfile(outDir, 'RBC_ButterHPF.png'));
imwrite(mat2gray(I_gauss_LP),  fullfile(outDir, 'RBC_GaussLPF.png'));
imwrite(mat2gray(I_gauss_HP),  fullfile(outDir, 'RBC_GaussHPF.png'));

% Compute spectrum and phase angle of Fourier transform.
% Select an image
I0 = imread(fullfile(inDir, 'RBC.jpg'));
if size(I0,3)==3, I = rgb2gray(I0); else, I = I0; end
I = im2double(I);

% Compute 2D Fourier Transform
F = fft2(I);

% Shift zero frequency to the center
Fshift = fftshift(F);

% Compute magnitude spectrum and phase angle
magnitude = log(1 + abs(Fshift));
phase = angle(Fshift);

% Display results
figure('Name','Spectrum and Phase','Color','w');
subplot(1,3,1); imshow(I, []); title('Original Image');
subplot(1,3,2); imshow(mat2gray(magnitude)); title('Magnitude Spectrum');
subplot(1,3,3); imshow(mat2gray(phase)); title('Phase Angle');

% Save images
imwrite(mat2gray(magnitude), fullfile(outDir, 'RBC_magnitude_spectrum.png'));
imwrite(mat2gray(phase), fullfile(outDir, 'RBC_phase_angle.png'));

% We applied the Fourier transform to one hand image (RBC.jpg)
% to visualize its magnitude and phase spectra.
% The magnitude spectrum shows dominant low-frequency components,
% while the phase contains the structure of the image.


% Reproduce image by combining phase angle and spectrum of two different image.
% Define  image paths
imgA = 'C:\Users\LHN10\Desktop\COMP 478\tutorial_week6\myHands\RBC.jpg';
imgB = 'C:\Users\LHN10\Desktop\COMP 478\tutorial_week6\myHands\LFC.jpg';
outDir = 'C:\Users\LHN10\Desktop\COMP 478\tutorial_week6';

% Read and convert to grayscale
A0 = im2double(rgb2gray(imread(imgA)));
B0 = im2double(rgb2gray(imread(imgB)));

% Resize to same size (important!)
[M, N] = size(A0);
B0 = imresize(B0, [M N]);

% Fourier transform
FA = fft2(A0);
FB = fft2(B0);

% Get magnitude and phase
magA = abs(FA);
magB = abs(FB);
phaseA = angle(FA);
phaseB = angle(FB);

% Combine them
F_AB = magA .* exp(1i * phaseB); 
F_BA = magB .* exp(1i * phaseA);

% Reconstruct images
img_AB = real(ifft2(F_AB));
img_BA = real(ifft2(F_BA));

% Display results
figure('Name','Combine Magnitude & Phase','Color','w');
subplot(2,3,1); imshow(A0, []); title('Image A (RBC)');
subplot(2,3,2); imshow(B0, []); title('Image B (LFC)');
subplot(2,3,4); imshow(img_AB, []); title('A magnitude + B phase');
subplot(2,3,5); imshow(img_BA, []); title('B magnitude + A phase');

% Save results
imwrite(mat2gray(img_AB), fullfile(outDir, 'A_mag_B_phase.png'));
imwrite(mat2gray(img_BA), fullfile(outDir, 'B_mag_A_phase.png'));


% Shift Fourier transform of image using property 4 (-1)^(x+y)
% Read image
I0 = imread("C:\Users\LHN10\Desktop\COMP 478\tutorial_week6\myHands\RBC.jpg");
if size(I0,3)==3, I = rgb2gray(I0); else, I = I0; end
I = im2double(I);

% Apply (-1)^(x+y)
[M, N] = size(I);
[x, y] = meshgrid(0:N-1, 0:M-1);
T = I .* ((-1).^(x + y));  

% Compute FFT
F1 = fft2(I);           
F2 = fft2(T);           

% Show comparison
figure('Name','Fourier Shift using (-1)^(x+y)','Color','w');
subplot(1,2,1);
imshow(log(1+abs(fftshift(F1))), []);
title('Without Shift');

subplot(1,2,2);
imshow(log(1+abs(F2)), []);
title('With Shift using (-1)^{(x+y)}');

% save result
outDir = "C:\Users\LHN10\Desktop\COMP 478\tutorial_week6";
exportgraphics(gcf, fullfile(outDir, 'Shift_Fourier_Property4.png'), 'Resolution', 200);


%  Apply Laplacian in frequency domain.
I0 = imread(fullfile(inDir, "RBC.jpg"));
if size(I0,3)==3, I = rgb2gray(I0); else, I = I0; end
I = im2double(I);
[M,N] = size(I);

% calculates the frequency coordinate grid
[u,v] = meshgrid(-N/2:N/2-1, -M/2:M/2-1);
D2 = (u.^2 + v.^2);

% 2️Define Laplace filter
H_lap = -4 * (pi^2) * D2; 

% Fourier transform
F = fftshift(fft2(I));

% Applied Laplace Filters
G = H_lap .* F;

% Turn back to the space domain
I_lap = real(ifft2(ifftshift(G)));

%  Normalisation for display purposes
I_lap_norm = mat2gray(I_lap);

% Show Results
figure('Name','Laplacian in Frequency Domain');
subplot(1,3,1); imshow(I,[]); title('Original Image');
subplot(1,3,2); imshow(log(1+abs(fftshift(F))),[]); title('FFT Spectrum');
subplot(1,3,3); imshow(I_lap_norm,[]); title('After Laplacian Filter');

% 8Save Results
imwrite(I_lap_norm, fullfile(outDir, 'Laplacian_FreqDomain.png'));
