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
We use Jaffe and CK+ as the dataset.
 - Jaffe
   - Michael J. Lyons, Shigeru Akemastu, Miyuki Kamachi, Jiro Gyoba. Coding Facial Expressions with Gabor Wavelets, 3rd IEEE International Conference on Automatic Face and Gesture Recognition, pp. 200-205 (1998).
 - CK+
   - Lucey P, Cohn J F, Kanade T, et al. The extended cohn-kanade dataset (ck+): A complete dataset for action unit and emotion-specified expression[C]//2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition-Workshops. IEEE, 2010: 94-101.



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
- CNN
  - For feature extraction + classification


## Network design
Using the classic convolutional neural network, the construction of the model mainly refers to a CVPR paper in 2018: A Compact Deep Learning Model for Robust Facial Expression Recognition to design the following network structure. After the input layer, a (1,1) convolutional layer is added to increase the nonlinear representation and the model level is shallow. , with fewer parameters (a large number of parameters are concentrated in the fully connected layer).
<div align="center"><img src="./assets/CNN.png" /></div>
<div align="center"><img src="./assets/model.png" /></div>


## Model training
The CNN model is mainly trained on JAFFE and CK+. JAFFE gives a half-length image, so face detection has to be done. The 5-fold cross-validation reached an accuracy rate of about 99%.

Use the command below to train the specified epochs on the dataset (jaffe or ck+）using a specific batch_size. The training will generate the corresponding visual training process. Here is an example of plotting the training process after the model has been trained on jaffe.

```shell
python src/train.py --dataset jaffe --epochs 300 --batch_size 32 
```
![](./assets/his_acc_jaffe.png)
![](./assets/his_loss_jaffe.png)



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
