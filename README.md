# CNN_ImageRetrieval
CNN model based on tensorflow for image classification and retrieval
## 功能说明
1、CNN_model.py中实现了一个2卷积层、2池化层和2全连接层的神经网络，并最后通过sigmoid函数做归一化处理。
2、image_preprocessing.py中进行了图片预处理。
3、model_training.py中用image_preprocessing.py提取的数据feed CNN_model.py中的神经网络进行训练。
4、image_test.py对于训练好的神经网络进行准确度测试。
## 使用说明
放入图片名列表文件imagelist.txt以及数据集./image/image/，先通过model_training.py进行模型训练，再通过image_test.py进行检索测试。
