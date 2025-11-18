function IMG = LOG_EdgeDetector(X, sigma, thr_inv)
% LOG_EdgeDetector  Laplacian of Gaussian (Marr–Hildreth) edge detector
% Usage:
%   IMG = LOG_EdgeDetector('plant4.jpg', 1.7759, 800);
%
% Inputs:
%   X        : image filename OR an image matrix
%   sigma    : Gaussian sigma (e.g., 1.7759)
%   thr_inv  : "inverse of threshold close to zero".
%              If >=1, threshold = 1/thr_inv; otherwise threshold = thr_inv.
%              Threshold is applied on normalized LoG response in [0,1].
%
% Output:
%   IMG      : logical edge map (1 = edge, 0 = non-edge)

% Read & prep image
 if ischar(X) || isstring(X)
        I = imread(X);
    else
        I = X;
    end

    if ndims(I) == 3
        I = rgb2gray(I);
    end
    I = im2double(I);
    
    % Build LoG kernel
    h = makeLoGKernel(sigma);

    % Convolution
    R = conv2(I, h, 'same');
    maxAbs = max(abs(R(:))) + eps;
    Rn = R / maxAbs;

    % Zero-crossing with magnitude test
    if nargin < 3 || isempty(thr_inv), thr_inv = 800; end
    if thr_inv >= 1
        tau = 1 / thr_inv;
    else
        tau = thr_inv; 
    end
    tau = max(min(tau, 0.2), 1e-6); 

    IMG = zeroCrossing2Dir(Rn, tau);
end

% Helpers
function h = makeLoGKernel(sigma)
%   h(x,y) = -(((x^2 + y^2) - sigma^2) / sigma^4) * exp(-(x^2 + y^2)/(2*sigma^2))

    radius = max(1, ceil(3*sigma));
    [x, y] = meshgrid(-radius:radius, -radius:radius);
    r2 = x.^2 + y.^2;

    h = -((r2 - sigma^2) / (sigma^4)) .* exp(-(r2) / (2*sigma^2));

    h = h - mean(h(:));
end

function E = zeroCrossing2Dir(Rn, tau)
    % neighbors (shifted copies)
    R_right = [Rn(:,2:end), Rn(:,end)];
    R_down  = [Rn(2:end,:);  Rn(end,:)];
    R_rd    = [Rn(2:end,2:end), Rn(2:end,end); Rn(end,2:end), Rn(end,end)];
    R_ld    = [Rn(2:end,1), Rn(2:end,1:end-1); Rn(end,1), Rn(end,1:end-1)];

    % sign change booleans
    sc_h  = (sign(Rn) .* sign(R_right) == -1) & (abs(Rn - R_right) > tau);
    sc_v  = (sign(Rn) .* sign(R_down)  == -1) & (abs(Rn - R_down)  > tau);
    sc_d1 = (sign(Rn) .* sign(R_rd)    == -1) & (abs(Rn - R_rd)    > tau);
    sc_d2 = (sign(Rn) .* sign(R_ld)    == -1) & (abs(Rn - R_ld)    > tau);

    % count directions with sign change
    count = double(sc_h) + double(sc_v) + double(sc_d1) + double(sc_d2);

    % keep pixels with >=2 directional sign-changes (per lecture hint)
    E = count >= 2;

    % Clean borders (optional): zero out the last row/col artifacts
    E(end,:) = false; E(:,end) = false;
end

IMG = LOG_EdgeDetector('plant4.jpg', 1.7759, 800);
imshow(IMG); title('LoG Edges');


% Exercise #2
IMG1 = LOG_EdgeDetector('LBC.jpg', 1.5, 600);
IMG2 = LOG_EdgeDetector('LFO.jpg', 2.5, 800);
IMG3 = LOG_EdgeDetector('RBO.jpg', 3.0, 1200);

figure;
subplot(1,3,1); imshow(IMG1); title('LBC: σ=1.5, thrInv=600');
subplot(1,3,2); imshow(IMG2); title('LFO: σ=2.5, thrInv=800');
subplot(1,3,3); imshow(IMG3); title('RBO: σ=3.0, thrInv=1200');

% Dataset: 
% 3 hand images (LBC, LFO, RBO) processed with LOG_EdgeDetector() 
% using σ = 1.5 / 2.5 / 3.0 and thrInv = 600 – 1200.

% Observation:
% 1. For all three images, the edge maps contain excessive white pixels, 
%    indicating too many detected edges.

% 2. This happens because the threshold was too low (1/600 ≈ 0.0017 – 1/1200 ≈ 0.0008), 
% so even small intensity changes triggered zero-crossings.

% 3. Increasing σ from 1.5 to 3.0 slightly smooths the result, 
% but background noise remains strong.

% Conclusion:
% 1. The current parameters are too sensitive for smooth surfaces such as hands.

% 2. To obtain cleaner contours, σ should be increased (e.g., 4–5) and/or thrInv set 
% higher (e.g., 2000–3000) to raise the threshold.

% 3. Pre-filtering each image with a small Gaussian 
% blur before LoG would further reduce noise.

