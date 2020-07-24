# Image Classification

## Motivation

In this section we will introduce the Image Classification problem, which is the task of assigning an input image one label from a fixed set of categories. This is one of the core problems in Computer Vision that, despite its simplicity, has a large variety of practical applications. Moreover, as we will see later in the course, many other seemingly distinct Computer Vision tasks (such as object detection, segmentation) can be reduced to image classification.

在本节中，我们将介绍图像分类问题，该问题是从一组固定的类别中为输入图像分配一个标签的任务。 这是Computer Vision的核心问题之一，尽管它很简单，却具有多种实际应用。 而且，正如我们稍后将在本课程中看到的那样，可以将许多其他看似不同的Computer Vision任务（例如对象检测，分割）简化为图像分类。

## Example

 For example, in the image below an image classification model takes a single image and assigns probabilities to 4 labels, *{cat, dog, hat, mug}*. As shown in the image, keep in mind that to a computer an image is represented as one large 3-dimensional array of numbers. In this example, the cat image is 248 pixels wide, 400 pixels tall, and has three color channels Red,Green,Blue (or RGB for short). Therefore, the image consists of 248 x 400 x 3 numbers, or a total of 297,600 numbers. Each number is an integer that ranges from 0 (black) to 255 (white). Our task is to turn this quarter of a million numbers into a single label, such as *“cat”*.

例如，在下面的图像中，图像分类模型拍摄一张图像并将概率分配给4个标签* {猫，狗，帽子，杯子} *。 如图所示，请记住，对于计算机，图像表示为一个大型3维数字数组。 在此示例中，猫图像的宽度为248像素，高度为400像素，并具有红色，绿色，蓝色（简称RGB）三个颜色通道。 因此，图像由248 x 400 x 3个数字或总共297,600个数字组成。 每个数字都是一个整数，范围是0（黑色）到255（白色）。 我们的任务是将这一百万分之一的数字变成一个单独的标签，例如*“ cat” *。



![classifyExample](../img/CV/classifyExample.png)

The task in Image Classification is to predict a single label (or a distribution over labels as shown here to indicate our confidence) for a given image. Images are 3-dimensional arrays of integers from 0 to 255, of size Width x Height x 3. The 3 represents the three color channels Red, Green, Blue.图像分类中的任务是预测给定图像的单个标签（或此处所示的标签分布，以表示我们的信心）。 图像是从0到255的整数的3维数组，大小为宽x高x3。3表示红色，绿色，蓝色三个颜色通道。

## Challenges

Since this task of recognizing a visual concept (e.g. cat) is relatively trivial for a human to perform, it is worth considering the challenges involved from the perspective of a Computer Vision algorithm. As we present (an inexhaustive) list of challenges below, keep in mind the raw representation of images as a 3-D array of brightness values:

由于识别视觉概念（例如猫）的任务对于人类而言相对来说是微不足道的，因此值得从计算机视觉算法的角度考虑所涉及的挑战。 当我们呈现以下（不完整的）挑战列表时，请记住图像的原始表示形式为亮度值的3D数组：

- **Viewpoint variation**. A single instance of an object can be oriented in many ways with respect to the camera. **视点变化**。 对象的单个实例可以相对于相机以多种方式定向。
- **Scale variation**. Visual classes often exhibit variation in their size (size in the real world, not only in terms of their extent in the image). **规模变化**。 视觉类的大小通常会变化（现实世界中的大小，不仅是图像的大小）。
- **Deformation**. Many objects of interest are not rigid bodies and can be deformed in extreme ways.**形变**。 许多感兴趣的对象不是刚体，并且可能以极端方式变形。
- **Occlusion**. The objects of interest can be occluded. Sometimes only a small portion of an object (as little as few pixels) could be visible.咬合。 感兴趣的对象可以被遮挡。 有时只能看到对象的一小部分（少至几个像素）。
- **Illumination conditions**. The effects of illumination are drastic on the pixel level. **照明条件**。 照明的影响在像素级别上非常明显。
- **Background clutter**. The objects of interest may *blend* into their environment, making them hard to identify. **背景混乱**。 感兴趣的对象可能会“融合”到其环境中，从而使其难以识别。
- **Intra-class variation**. The classes of interest can often be relatively broad, such as *chair*. There are many different types of these objects, each with their own appearance. **类内变化**。 感兴趣的类别通常相对较宽，例如* chair *。 这些对象有许多不同的类型，每种都有各自的外观。

A good image classification model must be invariant to the cross product of all these variations, while simultaneously retaining sensitivity to the inter-class variations. 一个好的图像分类模型必须对所有这些变异的叉积不变，同时又要保持对类别间变异的敏感性。

## The image classification pipeline

 We’ve seen that the task in Image Classification is to take an array of pixels that represents a single image and assign a label to it. Our complete pipeline can be formalized as follows: 我们已经看到，图像分类的任务是获取代表单个图像的像素数组，并为其分配标签。 我们完整的管道可以如下形式化：

- **Input:** Our input consists of a set of *N* images, each labeled with one of *K* different classes. We refer to this data as the *training set*. **输入：**我们的输入由一组* N *张图像组成，每张图像都标记有* K *个不同的类别之一。 我们将此数据称为“训练集”。
- **Learning:** Our task is to use the training set to learn what every one of the classes looks like. We refer to this step as *training a classifier*, or *learning a model*. **学习：**我们的任务是使用训练集来学习每一堂课的模样。 我们将此步骤称为“训练分类器”或“学习模型”。
- **Evaluation:** In the end, we evaluate the quality of the classifier by asking it to predict labels for a new set of images that it has never seen before. We will then compare the true labels of these images to the ones predicted by the classifier. Intuitively, we’re hoping that a lot of the predictions match up with the true answers (which we call the *ground truth*). **评估：**最后，我们通过要求分类器预测从未见过的一组新图像的标签来评估分类器的质量。 然后，我们将这些图像的真实标签与分类器预测的标签进行比较。 凭直觉，我们希望很多预测与真实答案（我们称为“基本事实”）相符。



# Data-Driven Approach 

1. Collect a dataset of images and labels 
2. Use Machine Learning algorithms to train a classifier 
3. Evaluate the classifier on new images

# Distance Metric

**K-Nearest Neighbors: Distance Metric**

**L1**

L1 Manhattan distance：
$$
d_1(I_1,I_2)=\sum_P=|I_1^P-I_2^P|
$$
**L2**

L2 Euclidean distance
$$
d_2(I_1,I_2)=\sqrt {\sum_P(I_1^P-I_2^P)^2}
$$

## Hyperparameters

These are hyperparameters: choices about the algorithms themselves. 

Very problem-dependent. Must try them all out and see what works best.

In **Image classification** we start with a **training set** of images and labels, and must predict labels on the test set. 

The **K-Nearest Neighbors** classifier predicts labels based on the **K** nearest training examples

 Distance metric and K are **hyperparameters** 

Choose hyperparameters using the **validation set**; 

Only run on the test set once at the very end! 

Pixel distance is not very informative. 

# Linear Classifier

$$
f(x,W) = Wx+b
$$

#  Reference

[Image Classification](https://cs231n.github.io/classification/)

































