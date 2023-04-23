import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread("lenna.png",0)
plt.rcParams['font.sans-serif'] = ['Heiti TC']

height = img.shape[0]
width = img.shape[1]

#创建一幅图像， uint8是专门用于存储各种图像的（包括RGB，灰度图像等），范围是从0–255
result_1 = np.zeros((height, width), np.uint8)

# 线性
for i in range(height):
    for j in range(width):
        gray = -(img[i, j])+255
        result_1[i, j] = np.uint8(gray)

# 分段线性
result_2 = np.zeros((height, width), np.uint8)
result_2_1 = np.zeros((height, width), np.uint8)

for i in range(height):
    for j in range(width):
        if (img[i, j])+50>255:
            gray = 255
        else:
            gray = (img[i, j])+50
        result_2_1[i, j] = np.uint8(gray)

#非线性
img=np.double(img)
result_3=np.log10(img+1)
result_3= np.uint8(result_3*255/np.max(result_3))

# 显示图形
plt.figure(num='comparison')
titles = ['原图像', '线性','分段线性','非线性']
images = [img, result_1,result_2_1,result_3 ]
for i in range(4):
    plt.subplot(2,2, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()



