# matlab代码
# clc,clear,close all
# f=checkerboard(8);
# f=im2double(f);
# len=9;theta=0;
# PSF = fspecial('motion',len,theta);
# g = imfilter(f,PSF,'circular');
# noise=imnoise(zeros(size(f)),'gaussian',0,0.001);
# g = imnoise(zeros(size(f)),'gaussian',0,0.001);
# subplot(221),imshow(g,[]);%模糊加噪声后的图像
# title('模糊加噪声后的图像');
# frest1=deconvwnr(g,PSF);   %维纳滤波
# subplot(222),imshow(frest1,[]),title('维纳滤波后的结果');
# Sn = abs(fft2(noise)).^2;   %噪声功率谱
# nA=sum(Sn(:))/numel(noise);  %平均噪声功率谱
# Sf = abs(fft2(f)).^2;    %信号功率谱
# fA=sum(Sf(:))/numel(f);    %平均信号功率谱
# R = nA/fA;   %求信噪比
# frest2=deconvwnr(g,PSF,R);
# subplot(223),imshow(frest2,[]),title('使用常数比例的维纳滤波');
# NACORR = fftshift(real(ifft(Sn)));
# FACORR = fftshift(real(ifft(Sf)));
# frest3=deconvwnr(g,PSF,NACORR,FACORR);  %自相关后的维纳滤波
# subplot(224),imshow(frest3,[]),title('使用自相关函数的维纳滤波');
#
