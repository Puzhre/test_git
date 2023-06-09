import cv2
import matplotlib.pyplot as plt
import numpy as np

gray = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

img_dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(img_dft)
fimg = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

idft_shift = np.fft.ifftshift(dft_shift)
ifimg = cv2.idft(idft_shift)
ifimg = 20*np.log(cv2.magnitude(ifimg[:, :, 0], ifimg[:, :, 1]))
ifimg=np.abs(ifimg)

plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.subplot(131), plt.imshow(gray, 'gray'), plt.title('原图像')
plt.axis('off')
plt.subplot(132), plt.imshow(np.int8(fimg), 'gray'), plt.title('傅里叶变换')
plt.axis('off')
plt.subplot(133), plt.imshow(np.int8(ifimg), 'gray'), plt.title('傅里叶逆变换')
plt.axis('off')
plt.savefig('a3_1.png')
plt.show()

R=np.real(fimg)
I=np.imag(fimg)
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.subplot(221), plt.imshow(R, 'gray'), plt.title('real')
plt.axis('off')
plt.subplot(222), plt.imshow(I, 'gray'), plt.title('imaginary')
plt.axis('off')
plt.subplot(223), plt.imshow(gray, 'gray'), plt.title('orginal')
plt.axis('off')
plt.subplot(224), plt.imshow(cv2.idft(R), 'gray'), plt.title('phase=0')
plt.axis('off')
plt.savefig('a3_2.png')
plt.show()

magnitude_spectrum = np.log(np.abs(fimg))
e=10e-5
phase_spectrum = np.log(np.angle(fimg)*180/np.pi+e)
Picture_Restructure = np.fft.fft2(1.*np.exp((np.angle(gray)*1j)))

cv2.destroyAllWindows()
cv2.waitKey(1)
