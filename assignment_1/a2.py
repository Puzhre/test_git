import cv2
import math
import numpy as np

class Img:
    def __init__(self,image,rows,cols,center=[0,0]):
        self.src=image
        self.rows=rows
        self.cols=cols
        self.center=center

    def Move(self,delta_x,delta_y):
        self.transform=np.array([[1,0,delta_x],[0,1,delta_y],[0,0,1]])

    def Horizontal(self):
        self.transform=np.array([[1,0,0],[0,-1,self.cols-1],[0,0,1]])

    def Vertically(self):
        self.transform=np.array([[-1,0,self.rows-1],[0,1,0],[0,0,1]])

    def Rotate(self,beta):
        self.transform=np.array([[math.cos(beta),-math.sin(beta),0],
                                 [math.sin(beta), math.cos(beta),0],
                                 [0,0,1]])

    def Process(self):
        self.dst=np.zeros((self.rows,self.cols),dtype=np.uint8)
        for i in range(self.rows):
            for j in range(self.cols):
                src_pos=np.array([i-self.center[0],j-self.center[1],1])
                [x,y,z]=np.dot(self.transform,src_pos)
                x=int(x)+self.center[0]
                y=int(y)+self.center[1]

                if x>=self.rows or y>=self.cols or x<0 or y<0:
                    self.dst[i][j]=255
                else:
                    self.dst[i][j]=self.src[x][y]

if __name__=='__main__':
    src=cv2.imread('lenna.png',0)
    rows = src.shape[0]
    cols = src.shape[1]

    img1,img2,img3=Img(src,rows,cols),Img(src,rows,cols),Img(src,rows,cols,[256,256])
    img1.Move(-50,-50)
    img1.Process()
    img2.Vertically()
    img2.Process()
    img3.Rotate(-math.radians(60))
    img3.Process()

    cv2.imshow("src", src)
    cv2.imshow("img1", img1.dst)
    cv2.imshow("img2", img2.dst)
    cv2.imshow("img3", img3.dst)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



