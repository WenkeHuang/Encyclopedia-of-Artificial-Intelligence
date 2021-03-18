# Exploring Simple Siamese Representation Learning

## MOtivation

Siamese network是一种无监督视觉表征学习模型的常见结构。这些模型最大限度地提高了同一图像的两个放大部分之间的相似性。
Siamese network的所有输出都“崩溃”成一个常量。目前有几种防止Siamese network崩溃的策略：（1）Contrastive learning，例如SimCLR，排斥负对，吸引正对，负对排除了来自解空间的恒定输出；（2）Clustering，例如SwAV，避免恒定输出；（3）BYOL仅依赖正对，但在使用动量编码器的情况下不会Siamese network崩溃。
作者提出的simple Siamese不使用上述的任一方法（负样本对，大批量，动量编码器），也能有很好的表现。

## Pseudocode

```python
for x in loader:
	x1, x2 = aug(x), aug(x)
  z1, z2 = f(x1), f(x2)
  p1, p2 = h(z1), h(z2)
  L = D(p1,z2)/2+D(p2,z1)/2
  L.backward()
  update(f,h)
 
def D(p,z):
  z = z.detach()
  p = normalize(p,dim=1)
  z = normalize(z,dim=1)
  return -(p*z).sum(dim=1).mean()
```

## Method

$$
D(p_1,z_2)=-\frac{p_1}{||p_1||_2} \cdot \frac{z_2}{||z_2||_2}
$$

$$
\mathcal{L} = \frac{1}{2}D(p_1,z_2)+\frac{1}{2}D(p_2,z_1)
$$

$$
D(p_1,stopgrad(z_2))
$$

$$
\mathcal{L}=\frac{1}{2}D(p_1,stoppgrad(z_2))+\frac{1}{2}D(p_2,stoppgrad(z_1))
$$



## Relative Setting

相关设置
优化器：SGD进行预训练。学习率具有余弦衰减时间表（cosine decay schedule ），基础lr为0.05，学习率进行线性缩放为lr×BatchSize/256。BatchSize=512，使用8GPU训练。
encoder f：ResNet-50作为backbone进行100epochs，Projection MLP具有3层，隐藏层为2048-d，BN层应用于每个FC层，激活函数使用ReLU，输出层无激活函数。
predictor h：MLP有两层，输出FC不具有BN和ReLU，隐藏层有BN和ReLU，输入维度为2048，隐藏层维度为512。
在不使用标签的情况下对1000级ImageNet训练集[11]进行了无监督的预训练。通过对训练集中的冻结表示进行监督的线性分类器，然后在验证集中对其进行测试（这是一种常见协议），可以评估预训练表示的质量。