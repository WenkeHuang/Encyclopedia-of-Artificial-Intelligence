# Bag of freebies
改变培训策略，或者只会增加培训成本的方法，对测试不影响。

**数据扩充：**
1.  光度畸变：调整图像的亮度、对比度、色调、饱和度和噪声
2.  几何畸变：加入随机缩放、剪切、翻转和反旋转

**模拟对象遮挡：**
1. random erase，CutOut：可以**随机选择图像中的矩形区域，并填充一个随机的或互补的零值**
2. hide-and-seek、grid mask：随机或均匀地选择图像中的多个矩形区域，并将其全部替换为0
**feature map：**
DropOut、DropConnect和**DropBlock**。

**结合多幅图像进行数据扩充：**
 MixUp、CutMix 
 创造出了一种基于CutMix的Mosaic data augmentation

**Style Transfer GAN**

**解决类别不平衡：**
- hard negative example mining (只适用两阶段)
- online hard example mining (只适用两阶段)
- focal loss

**label smoothing**
标签松弛

**bbox：**
1. IoU_loss
2. IoU_loss
3. DIoU_loss
4. IoU_loss

​  **YOLOv4 - use：**
*CutMix and Mosaic data augmentation*、DropBlock regularization、    Class label smoothing、CIoU-loss、*CmBN*、*Self-Adversarial Training*、       *Eliminate grid sensitivity*、Using multiple anchors for a single ground       truth、Cosine annealing scheduler、**Optimal hyperparameters 通过遗传算法找到最优参数**、Random        training shapes。

# Bag of specials
只会增加少量推理成本但却能显著提高对象检测精度的plugin modules和post-processing methods

**enhance receptive field**：扩充接收域
SPP，ASPP，RFB

**attention module:**
1、Squeeze-and-Excitation (SE)：可以改善resnet50在分类任务上提高 1%精度，但是会增加GPU推理时间10%。SENet
2、Spatial Attention Module (SAM)：可以改善resnet50在分类任务上提高0.5%精度，并且不增加GPU推理时间。

 **feature integration：**
早期使用skip connection、hyper-column。随着FPN等多尺度方法的流 行，提出了许多融合不同特征金字塔的轻量级模型。SFAM、ASFF、BiFPN。    SFAM的主要思想是利用SE模块对多尺度拼接的特征图进行信道级配重权。    ASFF使用softmax作为点向水平重加权，然后添加不同尺度的特征映射。     BiFPN提出了多输入加权剩余连接来执行按比例加权的水平重加权，然后加入不  同比例的特征映射。

 **activation function：**
ReLU解决了tanh和sigmoid的梯度消失问题。                     
LReLU ， PReLU ， ReLU6 ，SELU， Swish ， hard-Swish ， **Mish 其中Swish和Mish都是连续可微的函数。**

**post-processing method**
nms：c·p
soft-nms：解决对象的遮挡问题
DIoU nms：将中心点分布信息添加到BBox筛选过程中

​  **YOLOv4 - use：**
Mish activation、CSP、MiWRC、SPP-block、SAM、PAN、DIoU-NMS


# Selection of architecture
LSVRC2012 (ImageNet)数据集上的分类任务，CSPResNext50要比CSPDarknet53好得多。然而，在COCO数据集上的检测任务，CSP+Darknet53比CSPResNext50更好。
- backbone：CSP+Darknet53
- additional module：SPP
- neck：PANet 
- head：YOLOv3 (anchor based)

# Additional improvements
为了使检测器更适合于单GPU上的训练，做了如下补充设计和改进:
1. 入了一种新的数据增强方法Mosaic和自对抗训练(SAT)
2. 在应用遗传算法的同时选择最优超参数
3. 修改了一些现有的方法，如：SAM，PAN，CmBN

# Detail of BoF
## CuMix and Mosaic data augmentation
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200507134836908.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200507135210899.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNDA5NDM4,size_16,color_FFFFFF,t_70#pic_center)
每张图片四个子图片组成然后标签很杂了。
## DropBlock regularization
为了提高泛化能力，直接Dropout整片区域
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200507135321283.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNDA5NDM4,size_16,color_FFFFFF,t_70#pic_center)

## Class label smoothing
[0,0,1]-->[0.01,0.01,0.98]
[...] (1-a)+a/n[1,1....](这里a=0.03 n=3）

## CIoU-Loss
IoU 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200507135744251.png#pic_center)
但是这个有个问题，会存在梯度消失的问题hh
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020050719122221.png#pic_center)
引入了GIoU和DIoU，后者的那个两者中心点距离的平方除以
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200507191441987.png#pic_center)
最后是CIoU
在DIoU的基础上加入了αv
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200507192039821.png#pic_center)
DIoU-NMS
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200507192157730.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNDA5NDM4,size_16,color_FFFFFF,t_70#pic_center)
## CmBN
BN Batch Normalization
CBN Cross-Iteration Bach Normalization 通过泰勒多项式去估计几个连续batch的统计参数：为了适配单GPU，低batch也可以有好结果

## Self-Adversarial Training
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200507193308108.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNDA5NDM4,size_16,color_FFFFFF,t_70#pic_center)
## Eliminate grid sensitivity
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020050719354527.png#pic_center)
## Cosine annealing scheduler 模拟余弦退火

一种改变学习率的方式
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200507193648102.png#pic_center)
## Using multiple anchors for a single ground truth
为单ground truth做多个锚点

# Detail of BoS
## Mish 激活函数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200507193859439.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNDA5NDM4,size_16,color_FFFFFF,t_70#pic_center)
## CSP
CSPNet可以大大减少计算量，提高推理速度和准确性

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200507194058762.png#pic_center)
## SPP
不管输入尺寸是怎样的，SPP层可以产生固定大小的输出，用于多尺度训练，改变池化层大小步长等等实现输出大小一样
## SAM-block
不仅在通道也在空间上加了注意力机制
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200507194428507.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNDA5NDM4,size_16,color_FFFFFF,t_70#pic_center)作者认为需要把注意力细化，提高效果。
## PAN
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200507194549468.png#pic_center)




> 段落引用