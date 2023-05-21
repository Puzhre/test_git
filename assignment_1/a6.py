import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

def DPCM(yBuffer, dBuffer, re, w, h, bitnum):
    # yBuffer raw buffer
    # dBuffer x buffer
    # rebuildBuffer reconstruct buffer
    x = 2 ** (8 - bitnum)
    y = 2 ** (9 - bitnum)
    flow_upper_bound = 2 ** bitnum - 1
    for i in range(0, h):
        prediction = 128
        pred_error = yBuffer[i * w] - prediction
        tmp = (pred_error + 128) // x
        dBuffer[i * w] = tmp
        inv_pred_error = dBuffer[i * w] * x - 128
        re[i * w] = inv_pred_error + prediction
        for j in range(1, w):
            prediction = re[i * w + j - 1]
            predErr = yBuffer[i * w + j] - prediction
            tmp = (predErr + 255) // y
            dBuffer[i * w + j] = tmp
            invPredErr = dBuffer[i * w + j] * y - 255
            re[i * w + j] = invPredErr + prediction


Img = cv.imread(r'lenna.png')
Img = cv.cvtColor(Img, cv.COLOR_BGR2GRAY)
h = Img.shape[0]
w = Img.shape[1]

dBuffer1 = np.zeros((h * w), dtype=np.uint8)
rebuildBuffer1 = np.zeros((h * w), dtype=np.uint8)
yBuffer1 = Img.reshape(h * w)

DPCM(yBuffer1, dBuffer1, rebuildBuffer1, w, h, 1)
dBuffer1 = dBuffer1.reshape(h, w)
rebuildBuffer1 = rebuildBuffer1.reshape(h, w)

dBuffer2 = np.zeros((h * w), dtype=np.uint8)
rebuildBuffer2 = np.zeros((h * w), dtype=np.uint8)
yBuffer2 = Img.reshape(h * w)

DPCM(yBuffer2, dBuffer2, rebuildBuffer2, w, h, 2)
dBuffer2 = dBuffer2.reshape(h, w)
rebuildBuffer2 = rebuildBuffer2.reshape(h, w)

dBuffer4 = np.zeros((h * w), dtype=np.uint8)
rebuildBuffer4 = np.zeros((h * w), dtype=np.uint8)
yBuffer4 = Img.reshape(h * w)

DPCM(yBuffer4, dBuffer4, rebuildBuffer4, w, h, 4)
dBuffer4 = dBuffer4.reshape(h, w)
rebuildBuffer4 = rebuildBuffer4.reshape(h, w)

dBuffer8 = np.zeros((h * w), dtype=np.uint8)
rebuildBuffer8 = np.zeros((h * w), dtype=np.uint8)
yBuffer8 = Img.reshape(h * w)

DPCM(yBuffer8, dBuffer8, rebuildBuffer8, w, h, 8)
dBuffer8 = dBuffer8.reshape(h, w)
rebuildBuffer8 = rebuildBuffer8.reshape(h, w)

def psnr(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean( (img1/512.0 - img2/512.0) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

print(psnr(Img,rebuildBuffer2))

