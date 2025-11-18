%% COMP478 Week5
inDir  = "C:\Users\LHN10\Desktop\COMP 478\tutorial_week5\myHands";
outDir = fullfile(inDir , "fft_results_min");
if ~exist(outDir,'dir'), mkdir(outDir); end
imgs = [dir(fullfile(inDir,'*.jpg')); dir(fullfile(inDir,'*.png'))];

% Set parameters
D0 = 50;

for k = 1:numel(imgs)
    name = imgs(k).name;
    I0 = imread(fullfile(inDir, name));
    if size(I0,3)==3, I = rgb2gray(I0); else, I = I0; end
    I = im2double(I);
    [M,N] = size(I);

    % Calculate the image average with fft2
    F = fft2(I);                                
    avg_val = real(F(1,1)) / (M*N);              
    fid = fopen(fullfile(outDir, [name '_avg.txt']), 'w');
    fprintf(fid, 'Average via DC/(MN): %.6f\n', avg_val);
    fclose(fid);

    % fftshift Results
    Fshift = fftshift(F);
    specImg = log(1 + abs(Fshift));
    imwrite(mat2gray(specImg), fullfile(outDir, [name '_fftshift.png']));

    % Ideal low-pass filtering
    [u,v] = meshgrid(-N/2:N/2-1, -M/2:M/2-1);
    H = double(sqrt(u.^2 + v.^2) <= D0);
    G = Fshift .* H;
    If_lp = real(ifft2(ifftshift(G)));
    imwrite(mat2gray(If_lp), fullfile(outDir, [name '_idealLP_D' num2str(D0) '.png']));

    % Two 1D FFTs are equivalent to a 2D FFT.
    F1 = fft(I, [], 1);                        
    F2 = fft(F1, [], 2);                         
    diff_norm = norm(F2(:) - F(:)) / max(1e-12, norm(F(:)));
    fid = fopen(fullfile(outDir, [name '_fft2_equiv.txt']), 'w');
    fprintf(fid, '||FFT1D(rows+cols)-FFT2||/||FFT2|| = %.3e\n', diff_norm);
    fclose(fid);

    % show result
    figure('Name', name);
    subplot(2,2,1); imshow(I, []); title(['Original - ' name]);
    subplot(2,2,2); imshow(mat2gray(specImg)); title('fftshift spectrum');
    subplot(2,2,3); imshow(mat2gray(If_lp)); title(['Ideal LPF (D0=' num2str(D0) ')']);
    subplot(2,2,4); text(0.05,0.5,sprintf('Average=%.4f\nEquiv diff=%.2e', avg_val, diff_norm),...
                        'FontSize',12); axis off
end

disp(" Results have been saved to: " + string(outDir));

% Observation:
% For all hand images, the average intensity is similar (around 0.55–0.58).
% The FFT spectra always show strong low-frequency components in the center.
% When the fingers are opened, more high-frequency components appear,
% while closed fingers produce smoother spectra.
% Left/right or front/back hands do not make big differences.
% After applying the ideal low-pass filter, the hand contour is preserved
% but fine details become blurred.
% Overall, only the finger position (open vs. closed) shows a significant effect.
