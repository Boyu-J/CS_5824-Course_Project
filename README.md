## Introduction
使用卷积神经网络构建整个系统，在尝试了Gabor、LBP等传统人脸特征提取方式基础上，深度模型效果显著。在FER2013、JAFFE和CK+三个表情识别数据集上进行模型评估。


## Environment setup
Based on Python3 and Keras2 (TensorFlow后端), 具体依赖安装如下 (conda虚拟环境 is recommended).
```shell script
git clone https://github.com/Boyu-J/CS_5824-Course_Project.git
cd FacialExpressionRecognition # !!!!改名!!!
conda create -n FER python=3.6 -y
conda activate FER
conda install cudatoolkit=10.1 -y
conda install cudnn=7.6.5 -y
pip install -r requirements.txt
```


## Data preparation
We use Jaffe and CK+ as the dataset.Source:


## Methodology
### **Traditional method**
- Data pre-process
	- Image smoothness
	- Face detection (HAAR classifier in opencv)
- Feature engineering
	- Extract facial feature
		- LBP
		- Gabor
- Classifier
	- SVM
### **Deep learning method**
- Face detection
	- HAAR classifier
	- MTCNN (perform better than HAAR)
- 卷积神经网络 CNN
  - 用于特征提取+分类


## Network design
使用经典的卷积神经网络，模型的构建主要参考2018年CVPR几篇论文以及谷歌的Going Deeper设计如下网络结构，输入层后加入(1,1)卷积层增加非线性表示且模型层次较浅，参数较少（大量参数集中在全连接层）。
<div align="center"><img src="./assets/CNN.png" /></div>
<div align="center"><img src="./assets/model.png" /></div>


## Model training
主要在FER2013、JAFFE、CK+上进行训练，JAFFE给出的是半身图因此做了人脸检测。最后在FER2013上Pub Test和Pri Test均达到67%左右准确率（该数据集爬虫采集存在标签错误、水印、动画图片等问题），JAFFE和CK+5折交叉验证均达到99%左右准确率（这两个数据集为实验室采集，较为准确标准）。

执行下面的命令将在指定的数据集（fer2013或jaffe或ck+）上按照指定的batch_size训练指定的轮次。训练会生成对应的可视化训练过程，下图为在三个数据集上训练过程的共同绘图。

```shell
python src/train.py --dataset fer2013 --epochs 100 --batch_size 32 
```
![](./assets/loss.png)


## Application 
与传统方法相比，卷积神经网络表现更好，使用该模型构建识别系统，提供**GUI界面和摄像头实时检测**（摄像必须保证补光足够）。预测时对一张图片进行水平翻转、偏转15度、平移等增广得到多个概率分布，将这些概率分布加权求和得到最后的概率分布，此时概率最大的作为标签（也就是使用了推理数据增强）。

### **Graphical User Interface**

Notice: **Graphical User Interface界面预测只显示最可能是人脸的那个脸表情，但是对所有检测到的人脸都会框定预测结果并在图片上标记，标记后的图片在output目录下。**

执行下面的命令即可打开GUI程序，该程序依赖PyQT设计，在一个测试图片上进行测试效果如下图。

```shell
python src/gui.py
```
![](./assets/gui.png)

上图的GUI反馈的同时，会对图片上每个人脸进行检测并表情识别，处理后如下图。

![](./assets/rst.png)
