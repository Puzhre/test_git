from scipy.signal import wiener
import cv2
import numpy as np
import matplotlib.pyplot as plt

def gasuss_noise(image, mean=0, var=0.001):
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out

if __name__ == '__main__':
    lena = cv2.imread(r'lenna.png')
    if lena.shape[-1] == 3:
        lenaGray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    else:
        lenaGray = lena.copy()

    lenaNoise = gasuss_noise(lenaGray)

    lenaNoise = lenaNoise.astype('float64')
    lenaWiener = wiener(lenaNoise, [3, 3])
    lenaWiener = np.uint8(lenaWiener / lenaWiener.max() * 255)
    
    plt.rcParams['font.sans-serif'] = ['Heiti TC']
    plt.subplot(131), plt.imshow(lenaGray, 'gray'), plt.title('原图')
    plt.axis('off')
    plt.subplot(132), plt.imshow(lenaNoise, 'gray'), plt.title('添加高斯噪声后的图像')
    plt.axis('off')
    plt.subplot(133), plt.imshow(lenaWiener, 'gray'), plt.title('经过维纳滤波后的图像')
    plt.axis('off')
    plt.savefig('a5_1.png')
    plt.show()


