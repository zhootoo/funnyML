# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 20:53:18 2017

@author: Zhanghongtao
"""
from PIL import Image
import numpy as np
import scipy.misc as misc
import pickle


def ImageToMatrixRGB(filename):
    """将图片转换成RGB矩阵
    filename : 图片地址
    return : 返回图片宽度*高度*通道数 X 1 的RGB array矩阵
    """
    im = misc.imread(filename)
    im = im / 255.0 #将像素归一化
    width, height, layer = im.shape
    im = im.reshape(width * height * layer)
    return im
##########################################################################
def ImageToMatrixWB(filename):
    """将图片转换成黑白然后转成矩阵
    filename : 图片地址
    return : 返回图片宽度*高度 X 1的array矩阵
    """
    im = Image.open(filename)
    # 显示图片 img.show
    width,height = im.size
    im = im.convert("L") 
    data = im.getdata()
    data = np.matrix(data,dtype='float')/255.0
    #new_data = np.reshape(data,(width,height))
    #new_data = np.reshape(data,(height,width))
    return data.reshape(width*height,1)
#######################################################################
def sigmod(inputs):
    """sigmod函数"""
    return [1 / (1 + np.exp(- x)) for x in inputs]
########################################################################
class BinaryClassfication(object):
    """
    实现二分类任务
    
    X : 二分类的属性数据，如果有m条数据，n个特征，则X为n*m的array矩阵
    Y : 标签数据为0或者1，传入1*m array矩阵
    train : 是否保留20%验证集，默认False
    alpha : 学习速率
    """
    def __init__(self, X, Y, train = False,alpha=0.001):
        self.X = X
        self.Y = Y
        self.alpha = alpha
        self.train = train
        self.n = 1200 #n个特征
        self.m = 20 #m条数据
        self.W = np.zeros([self.n, 1]) #权重矩阵n*1,初始化为0
        self.b = 0 #截距
        self.cost=[]
        #用于保存每次迭代产生的W和b
        self.W_save = np.zeros([1200,300])
        self.b_save = np.zeros(300)
    
    def fit(self, times=1):
        """开始拟合
        times指定迭代次数"""
        for i in range(times):
            print('###############'+str(i+1)+'#################')
            Z = np.dot(self.W.T, self.X) + self.b
            #print(Z.shape)
            A = sigmod(Z)
            
            dz = A - self.Y
            dw = np.dot((1/self.m * self.X), dz.T)
            db = 1/self.m * np.sum(dz)
            print('cost:'+str(db))
            #用于保存每次迭代的b,W,cost
            self.cost.append(db)
            self.W_save[:,i] = self.W.reshape(1200)
            self.b_save[i] = self.b
            self.W = self.W - self.alpha * dw #更新W矩阵
            self.b -= self.alpha * db #更新截距
            
    
    def predict(self, X):
        """预测函数
        X : n*m array矩阵
        return : 最后预测为1类别的概率
        """
        return sigmod(np.dot(self.W.T , X) + self.b)
        
if __name__ == '__main__':
    #特征矩阵X为20*20*3 X 20  
    X = np.zeros([1200,20])
    #20张图片所以有20个标签
    Y = np.zeros([1,20])
    #读取图片初始化X,Y
    for i in range(0,10):
        #读入猫咪数据
        #np.c_[X, ImageToMatrixRGB('D:/Workspace/github_blog/funny_ML/al_cat/'+'cat'+str(i+1)+'.jpg')]
        X[:,i] = ImageToMatrixRGB('D:/Workspace/github_blog/funny_ML/al_cat/'+'cat'+str(i+1)+'.jpg')
        #猫咪标签为1
        Y[0,i] = 1
    for i in range(10,20):
        #读入阿狸数据
        #np.c_[X, ImageToMatrixRGB('D:/Workspace/github_blog/funny_ML/al_cat/'+'al'+str(i-9)+'.jpg')]
         #阿狸标签为0
        X[:,i] = ImageToMatrixRGB('D:/Workspace/github_blog/funny_ML/al_cat/'+'al'+str(i-9)+'.jpg')
        Y[0,i] = 0
    bf = BinaryClassfication(X, Y)
    bf.fit(300)
    b_save = bf.b_save
    cost_save = bf.cost
    W_save = bf.W_save
    
    
        

    
        
        
            
            
            
        
        