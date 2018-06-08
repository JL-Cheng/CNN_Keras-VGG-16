# coding=gbk

from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras import backend

import matplotlib.pyplot as plt
from PIL import Image
import h5py as h5py
import numpy as np
import tensorflow as tf

def get_one_image(img_dir):
    x=[]
    image=Image.open(img_dir)
    image=image.resize([400,300])
    x.append(np.array(image))
    x=np.array(x)

    return x

def images_show(images):
    images_dir='./image/image/'
    plt.subplots(num='result pictures window',figsize=(8,6))
    for i in range(10):
        image=Image.open(images_dir+images[i][0])
        image=image.resize([image.width*5,image.height*5])
        plt.subplot(2,5,1+i)
        plt.axis('off')
        plt.imshow(image)
    plt.suptitle('result pictures')
    plt.show()
    
def test():
    images_dir='./image/image/'
    log_dir="./VGG_log/VGG_model.h5"

    images_cat=open("imagelist.txt")
    #保存所有图像经过模型计算之后的数组
    images_tested=[]

    #重载模型
    model = load_model(log_dir)

    for line in images_cat.readlines():
        image_name=line.strip('\n')
        image_array=get_one_image(images_dir+image_name)               
  
        prediction=model.predict(image_array)
        prediction=np.array(prediction,dtype='float32')
        images_tested.append([image_name,prediction])

        print(image_name)
        print(prediction)
        
        #测试单张图片
    while (True):
            test_file=input('输入测试图片:')
            if(test_file=='z'):
                break

            image_name=test_file
            image_array=get_one_image(images_dir+image_name)

            prediction=model.predict(image_array)
            prediction=np.array(prediction,dtype='float32')
            test_result=[]
            for sample in images_tested:
                distance=np.sqrt(np.sum(np.square(sample[1]-prediction)));
                distance.astype('float32')
                test_result.append([sample[0],distance])
                                
            #将结果排序
            test_result=np.array(test_result)
            test_result=test_result[np.lexsort(test_result.T)]
            for i in range(10):
                print(test_result[i][0])

            images_show(test_result)

test()