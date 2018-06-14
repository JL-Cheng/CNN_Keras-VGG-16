# CNN & Keras-VGG-16 
CNN model based on tensorflow and Modified 'VGG-16' model based on Keras for image classification and retrieval.
## 功能说明
1、CNN_model.py中实现了一个2卷积层、2池化层和2全连接层的神经网络，并最后通过sigmoid函数做归一化处理。<br>
2、image_preprocessing.py中进行了图片预处理。<br>
3、CNN_model_training.py中用image_preprocessing.py提取的数据feed CNN_model.py中的神经网络进行训练。<br>
4、image_test_CNN.py对于训练好的CNN神经网络进行准确度测试。<br>
5、Keras_VGG16_model.py中基于Keras的VGG-16模型实现了自定义的神经网络。<br>
6、image_test_VGG.py中对于训练好的VGG-16模型进行准确度测试。<br>
## 使用说明
### CNN_model
放入图片名列表文件imagelist.txt以及数据集./image/image/，先通过CNN_model_training.py进行模型训练，再通过image_test_CNN.py进行检索测试。<br>
### VGG-16_model
放入图片名列表文件imagelist.txt以及数据集./image/image/，先通过Keras_VGG16_model.py进行模型训练，再通过image_test_VGG.py进行检索测试。<br>
