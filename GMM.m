clc
clear all
load('inputdata_wine.mat');
load('targetdata_wine.mat');
load('matlab.mat');
% x1(1,:) = x(1,:)/15;
% x1(2,:)= x(2,:)/6;
% x1(3,:)= x(3,:)/3;
% x1(4,:) = x(4,:)/30;
% x1(5,:) = x(5,:)/130;
% x1(6,:) = x(6,:)/4;
% x1(7,:) = x(7,:)/4;
% x1(8,:) = x(8,:)/0.6;
% x1(9,:) = x(9,:)/3;
% x1(10,:) = x(10,:)/12;
% x1(11,:) = x(11,:)/2;
% x1(12,:) = x(12,:)/4;
% x1(13,:) = x(13,:)/1500;
x1 = x;
% MU = mean(x1,2); 
% MU2 = max(x1,[],2);
% MU3 = min(x1,[],2);


INITIALMEAN1 = repmat(minmidmax(:,3),1,178);
INITIALMEAN2 = repmat(minmidmax(:,1),1,178);
INITIALMEAN3 = repmat(minmidmax(:,2),1,178);
SIGMA1 = ((x1-INITIALMEAN1)*(x1-INITIALMEAN1)')/(size(x1,2));
SIGMA2 = ((x1-INITIALMEAN2)*(x1-INITIALMEAN2)')/(size(x1,2));
SIGMA3 = ((x1-INITIALMEAN3)*(x1-INITIALMEAN3)')/(size(x1,2));
% INITIALSIGMA1 = repmat(SIGMA1,1,1,178);
INITIALGAUSS1 = mvnpdf(x1',INITIALMEAN1',SIGMA1); 
INITIALGAUSS2 = mvnpdf(x1',INITIALMEAN2',SIGMA2);
INITIALGAUSS3 = mvnpdf(x1',INITIALMEAN3',SIGMA3);
W1 = 0.2; W2 = 0.3; W3 = 0.5;
Newgauss1 = INITIALGAUSS1; Newgauss2 = INITIALGAUSS2; Newgauss3 = INITIALGAUSS3;
for i = 1:50
PostNumer = [W1*Newgauss1, W2*Newgauss2, W3*Newgauss3];
PostDen = sum(PostNumer,2); 
R = [PostNumer(:,1)./PostDen, PostNumer(:,2)./PostDen, PostNumer(:,3)./PostDen];
m1 = sum(R(:,1));
mu1 = x1*R(:,1)/m1;
m2 = sum(R(:,2));
mu2 = x1*R(:,2)/m2;
m3 = sum(R(:,3));
mu3 = x1*R(:,3)/m3;
sigma1 = (repmat(R(:,1),1,13))'.*(x1-repmat(mu1,1,178))*(x1-repmat(mu1,1,178))'/m1;
sigma2 = (repmat(R(:,2),1,13))'.*(x1-repmat(mu2,1,178))*(x1-repmat(mu2,1,178))'/m2;
sigma3 = (repmat(R(:,3),1,13))'.*(x1-repmat(mu3,1,178))*(x1-repmat(mu3,1,178))'/m3;
W1 = m1/size(x1,2);
W2 = m2/size(x1,2);
W3 = m3/size(x1,2);
Newgauss1 = mvnpdf(x1',mu1',sigma1'); % DISCREPANCY??
Newgauss2 = mvnpdf(x1',mu2',sigma2'); % DISCREPANCY??
Newgauss3 = mvnpdf(x1',mu3',sigma3'); % DISCREPANCY??
end
Gauss = [Newgauss1 Newgauss2 Newgauss3];
R;
A = (R>0.4);
plotconfusion(t,A');