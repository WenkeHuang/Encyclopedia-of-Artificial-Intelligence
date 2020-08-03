# 背景

**YoloV4-Tiny是YoloV4的简化版，少了一些结构，但是速度大大增加了，YoloV4共有约6000万参数，YoloV4-Tiny则只有600万参数。**

**YoloV4-Tiny仅使用了两个特征层进行分类与回归预测。**

# 结构分析

## 主干特征提取网络Backbone

当输入是416x416时，特征结构如下：

<img src="../../img/Paper/ObjectDetection/YOLOv4_Backbone_416.png" alt="YOLOv4_Backbone_416" style="zoom:80%;" />

当输入是608x608时，特征结构如下：

<img src="../../img/Paper/ObjectDetection/YOLOv4_Backbone_608.png" alt="YOLOv4_Backbone_608" style="zoom:80%;" />

而在YoloV4-Tiny中，其使用了CSPdarknet53_tiny作为主干特征提取网络。
和CSPdarknet53相比，**为了更快速，将激活函数重新修改为LeakyReLU**。

**利用主干特征提取网络，我们可以获得两个shape的有效特征层，即CSPdarknet53_tiny最后两个shape的有效特征层，传入加强特征提取网络当中进行FPN的构建。**

## 特征金字塔

YoloV4-Tiny中使用了FPN的结构，主要是对第一步获得的两个有效特征层进行特征融合。

**FPN会将最后一个shape的有效特征层卷积后进行上采样，然后与上一个shape的有效特征层进行堆叠并卷积。**

## YoloHead利用获得到的特征进行预测

在特征利用部分，YoloV4-Tiny提取**多特征层进行目标检测**，一共**提取两个特征层**，两个特征层的shape分别为(38,38,128)、(19,19,512)。

输出层的shape分别为(**19,19,75**)，(**38,38,75**)，**最后一个维度为75是因为该图是基于voc数据集的，它的类为20种，YoloV4-Tiny只有针对每一个特征层存在3个先验框，所以最后维度为3x25；
如果使用的是coco训练集，类则为80种，最后的维度应该为255 = 3x85**，两个特征层的shape为(**19,19,255**)，(**38,38,255**)























































