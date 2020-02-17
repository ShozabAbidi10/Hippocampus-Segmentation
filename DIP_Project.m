%% Reading the Image
imgscan = imread('sample1.JPG');
imgscan = rgb2gray(imgscan);
% Cropping to the relevant region
imgscan1 = imgscan(170:259,195:284);
figure, subplot(121), imshow(imgscan);
subplot(122), imshow(imgscan1);

%% preprocessing
equalised = imgscan1;
[m, n] = size(imgscan1);
for i=1:floor(m/8)
    for j=1:floor(n/8)
        equalised((i-1)*8+1:i*8,(j-1)*8+1:j*8) = adapthisteq(equalised((i-1)*8+1:i*8,(j-1)*8+1:j*8),'clipLimit',0.001,'Distribution','rayleigh');% = histeq((i-1)*8+1:i*8,(j-1)*8+1:j*8);
        equalised((i-1)*8+1:i*8,(j-1)*8+1:j*8) = imresize(equalised((i-1)*8+1:i*8,(j-1)*8+1:j*8), [8 8], 'bilinear');
    end
end
equalised = imresize(equalised, [m n], 'bilinear');
%imshow(equalised);
%% masking  
marker = imerode(equalised, strel('disk', 1));
masked = imreconstruct(marker, imgscan1);
masked = imadjust(masked, [0.1 0.5], [0 1], 0.4);
figure,imshow(masked);
%% Only region growing on the image after it has been preprocessed 
thr = 7;                  % setting threshold region growing
seedpoints = zeros(90);   % setting up seedpoints array
seedpoints(40,40) = 1;     % setting up the seedpoint at the centre
%figure, imshow(seedpoints);
[seg_weld, no_of_regions, seed_points_img, thr_test_img] = regiongrow(masked, seedpoints, thr);   % applying region growing
%figure,imshow(seed_points_img);
%figure,imshow(thr_test_img);
%figure,imshow(seg_weld); % segmented region
% over laying segmented region on the original image
regiongrown = imgscan1;   
regiongrown(seg_weld==1) = 255;                   
figure, imshow(regiongrown);
% gettinng the boundry of the segmented region and overlaying it on the
% original image
eroded_seg = imerode(seg_weld,strel('square',3));
figure, imshow(eroded_seg);
boundary = seg_weld - eroded_seg;                 
figure, imshow(boundary);
regiongrown_bound = imgscan1;                      
regiongrown_bound(boundary == 1) = 255;           % 
figure, imshow(regiongrown_bound);
% Getting the convex hull of the segmented region and overlaying it on the
% original image
conv_hull = bwconvhull(seg_weld);
figure, imshow(conv_hull);
regiongrown_polygon = imgscan1;
regiongrown_polygon(conv_hull == 1) = 255;
figure, imshow(regiongrown_polygon)
% Getting the boundary of the convex hull of the segmented region and
% overlaying it on the original image
eroded_seg = imerode(conv_hull,strel('square',3));
figure, imshow(eroded_seg)
boundary = conv_hull - eroded_seg;
figure, imshow(boundary)
regiongrown_bound = imgscan1;
regiongrown_bound(boundary == 1) = 255;
figure, imshow(regiongrown_bound); % xlabel("resulted segmented image from region growing method)");
%% Only using Level Set on original image
% Li's Algorithm is used
imgscan = imread('sample1.JPG');
imgscan = rgb2gray(imgscan);
%imgscan1 = imgscan(160:259,185:284);
imgscan1 = imgscan(170:259,195:284);
figure, imshow(imgscan1);
Img = double(imgscan1);
%% parameter setting for energy equation:
timestep=1;  % time step
mu=0.2/timestep;  % coefficient of the distance regularization term R(phi)
% iter_inner=5;
% iter_outer=20;
% lambda=5; % coefficient of the weighted length term L(phi)
% alfa=5;  % coefficient of the weighted area term A(phi)
% epsilon=-3; % papramater that specifies the width of the DiracDelta function

iter_inner=10;
iter_outer=20;
lambda=5; % coefficient of the weighted length term L(phi)
alfa=-3;  % coefficient of the weighted area term A(phi)
epsilon=1.5; % papramater that specifies the width of the DiracDelta function

sigma=.8;    % scale parameter in Gaussian kernel
G=fspecial('gaussian',15,sigma); % Caussian kernel
Img_smooth=conv2(Img,G,'same');  % smooth image by Gaussiin convolution
figure, imshow(Img_smooth);
[Ix,Iy]=gradient(Img_smooth);
f=Ix.^2+Iy.^2;
g=1./(1+f);  % edge indicator function.

% initialize LSF as binary step function
c0=2;
initialLSF = c0*ones(size(Img));
figure, imshow(initialLSF);
% generate the initial region R0 as two rectangles
initialLSF(32:37,30:35)=-c0; 
initialLSF(40:50,45:55)=-c0;
% initialLSF(40:44,41:47)=-c0; 
% initialLSF(55:65,57:67)=-c0;
phi=initialLSF;
figure, imshow(phi);

figure(1);xlabel('')
mesh(-phi);   % for a better view, the LSF is displayed upside down
hold on;  contour(phi, [0,0], 'r','LineWidth',2);
title('Initial level set function');
view([-80 35]);

figure(2);
imagesc(Img,[0, 255]); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r');
title("resulted segmented image from level set method");
pause(0.5);

potential=2;  
if potential ==1
    potentialFunction = 'single-well';  % use single well potential p1(s)=0.5*(s-1)^2, which is good for region-based model 
elseif potential == 2
    potentialFunction = 'double-well';  % use double-well potential in Eq. (16), which is good for both edge and region based models
else
    potentialFunction = 'double-well';  % default choice of potential function
end  

% start level set evolution
for n=1:iter_outer
    phi = drlse_edge(phi, g, lambda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction);    
    if mod(n,2)==0
        figure(2);
        imagesc(Img,[0, 255]); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r');
    end
end

% refine the zero level contour by further level set evolution with alfa=0
alfa=0;
iter_refine = 10;
phi = drlse_edge(phi, g, lambda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction);

finalLSF=phi;
%% Level Set Method using the result from Region Growing as prior information
% The method proposed in the paper
% Li's Algorithm is used
imgscan = imread('sample1.JPG');
imgscan = rgb2gray(imgscan);
%imgscan1 = imgscan(160:259,185:284);
imgscan1 = imgscan(170:259,195:284);
figure, imshow(imgscan1);
Img=double(imgscan1);
figure, imshow(Img);
%% parameter setting
timestep=1;  % time step
mu=0.2/timestep;  % coefficient of the distance regularization term R(phi)
iter_inner=20;
iter_outer=10;
lambda=5; % coefficient of the weighted length term L(phi)
alfa=-3;  % coefficient of the weighted area term A(phi)
epsilon=1.5; % papramater that specifies the width of the DiracDelta function

sigma=.8;    % scale parameter in Gaussian kernel
G=fspecial('gaussian',15,sigma); % Caussian kernel
Img_smooth=conv2(Img,G,'same');  % smooth image by Gaussiin convolution
[Ix,Iy]=gradient(Img_smooth);
f=Ix.^2+Iy.^2;
g=1./(1+f);  % edge indicator function.

% initialize LSF as binary step function
 c0=2;
%initialLSF = c0*ones(size(Img));
% % generate the initial region R0 as two rectangles
%  initialLSF(25:35,20:25)=-c0; 
%  initialLSF(25:35,40:50)=-c0;
% phi=initialLSF;
a = (-c0*eroded_seg);
phi_1 = c0*(~eroded_seg);
phi_1 = im2double(phi_1 + a);
imshow(phi_1);

figure(1);
mesh(-phi_1);   % for a better view, the LSF is displayed upside down
hold on;  contour(phi_1, [0,0], 'r','LineWidth',2);
title('Initial level set function');
view([-80 35]);

figure(2);
imagesc(Img,[0, 255]); axis off; axis equal; colormap(gray); hold on;  contour(phi_1, [0,0], 'r');
title('Initial zero level contour');
pause(0.5);

potential=2;  
if potential ==1
    potentialFunction = 'single-well';  % use single well potential p1(s)=0.5*(s-1)^2, which is good for region-based model 
elseif potential == 2
    potentialFunction = 'double-well';  % use double-well potential in Eq. (16), which is good for both edge and region based models
else
    potentialFunction = 'double-well';  % default choice of potential function
end  

% start level set evolution
for n=1:iter_outer
    phi_1 = drlse_edge(phi_1, g, lambda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction);    
    if mod(n,2)==0
        figure(2);
        imagesc(Img,[0, 255]); axis off; axis equal; colormap(gray); hold on;  contour(phi_1, [0,0], 'r');
    end
end

% refine the zero level contour by further level set evolution with alfa=0
alfa=0;
iter_refine = -20;
phi_1 = drlse_edge(phi_1, g, lambda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction);

finalLSF=phi_1;
figure(2);
imagesc(Img,[0, 255]); axis off; axis equal; colormap(gray); hold on;  contour(phi_1, [0,0], 'r');
hold on; % contour(phi, [0,0], 'r');
%str=['Final zero level contour, ', num2str(iter_outer*iter_inner+iter_refine), ' iterations'];
%title(str);

figure;
mesh(-finalLSF); % for a better view, the LSF is displayed upside down
hold on;  contour(phi_1, [0,0], 'r','LineWidth',2);
view([-80 35]);
str=['Final level set function, ', num2str(iter_outer*iter_inner+iter_refine), ' iterations'];
title(str);
axis on;
[nrow, ncol]=size(Img);
axis([1 ncol 1 nrow -5 5]);
set(gca,'ZTick',[-3:1:3]);
set(gca,'FontSize',14);