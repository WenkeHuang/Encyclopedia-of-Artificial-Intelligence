# Barlow Twins: Self-Supervised Learning via Redundancy Reduction

## Motivation

自我监督学习（SSL）正在大型计算机视觉基准上采用监督方法迅速缩小差距。 SSL的成功方法是学习与输入样本的失真无关的表示形式。但是，这种方法经常出现的问题是琐碎的常量表示形式的存在。当前大多数方法都通过仔细的实现细节来避免这种崩溃的解决方案。我们提出一个目标函数，通过测量两个相同网络的输出之间的互相关矩阵，该互相关矩阵由样本的失真版本提供，并使其与单位矩阵尽可能接近，从而自然避免了此类崩溃。这使得样本的失真版本的表示向量相似，同时使这些向量的分量之间的冗余最小化。由于神经科学家H. Barlow的冗余减少原理应用于一对相同的网络，因此该方法称为BARLOW TWINS。 BARLOW TWINS不需要大批量，也不需要网络双胞胎之间的不对称性，例如预测器网络，梯度停止或权重更新的移动平均值。它允许使用非常高维的输出向量。 BARLOW TWINS优于ImageNet上用于低数据状态下的半监督分类的方法，并且与具有线性分类器头的ImageNet分类以及分类和对象检测的传输任务的当前技术水平相当

## Method

The distorted views are obtained via a distribution of data augmentation $\mathcal{T}$.

Get two batches of distorted views $Y^A$ and $Y^B$

A deep network with trainable parameters $\theta$

Produce a baches of representations $Z^A$ and $Z^B$


$$
\mathcal{L}_{BT} \triangleq \sum_i(1-\mathcal{C}_{ii})^2+\lambda \sum_i\sum_{j \neq i }\mathcal{C_{ij}^2}
$$
而对于具体的$C_{ij}$
$$
C_{ij} \triangleq \sum \frac{\sum_b z^A_{b,i} z^B_{b,j}} {\sqrt{\sum_b (x_{b,i}^A)^2} \sqrt{\sum_b (x_{b,j}^B)^2}}
$$












