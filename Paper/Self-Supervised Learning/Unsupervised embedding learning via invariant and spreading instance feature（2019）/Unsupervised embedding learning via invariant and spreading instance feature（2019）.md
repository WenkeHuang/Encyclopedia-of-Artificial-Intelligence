# Unsupervised embedding learning via invariant and spreading instance feature

## Motivation

同类别样本距离相近，从而会集中在一起；不同类别样本距离较大，从而会分散分布

## Methods

 **Data augmentation invariant**

对于数据增强后的样本，期望他能分类正确
$$
P(i|\widehat{x}_i)=\frac{exp(f_i^T \widehat{f}_i /\tau)}{\sum_{k=1}^m exp(f_k^T \widehat{f}_i /\tau )}
$$
**Instance spreadout features**

而对于其他负样本，我们期待他们分类错误：

此概率表述的是负样本被分类到和i样本一致的隐变量空间
$$
P(i|x_j)=\frac{exp(f_i^T  f_j /\tau)}{\sum_{k=1}^m exp(f_k^T f_j /\tau )} \ j \neq i
$$
因而所期望的是：
$$
P_i = P(i|\widehat{x}_i) \Pi_{j \neq i}(1-p(i|x_j))
$$
The negative log likelihood is given by:
$$
\mathcal{J}_i = -log P(i|\widehat{x}_i)-\sum_{j \neq i}log(1-P(i|x_j))
$$


