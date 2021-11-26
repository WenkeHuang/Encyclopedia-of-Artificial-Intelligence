#  Bootstrap your own latent A new approach to self-supervised Learning

## Motivation

这篇论文的motivation来源于一个有趣的实验，首先有一个网络参数随机初始化且固定的target network，target network的top1准确率只有1.4%，target network输出feature作为另一个叫online network的训练目标，等这个online network训练好之后，online network的top1准确率可以达到18.8%，这就非常有意思了，假如将target network替换为效果更好的网络参数（比如此时的online network），然后再迭代一次，也就是再训练一轮online network，去学习新的target network输出的feature，那效果应该是不断上升的，类似左右脚踩楼梯不断上升一样。BYOL基本上就是这样做的，并且取得了非常好的效果。

## Idea

人抬人
$$
\mathcal{L}_{\theta,\xi}  \triangleq ||\overline{q_{\theta}}(z_{\theta})-\overline{z}^,_\xi)||_2^2 = 2-2 \cdot \frac{	q_{\theta}<(z_\theta),z^,_\xi>}{||q_\theta (z_\theta)||_2 \cdot ||z_\xi^,||_2}
$$
Update:
$$
\theta \leftarrow optimizer(\theta,\bigtriangledown_\theta \mathcal{L}^{BYOL}_{\theta,\xi},\eta)
$$

$$
\xi \leftarrow \tau \xi + (1-\tau)\theta
$$



##  Result

实验结果方面，linear evaluation（特征提取网络的参数不变，仅训练新增的一个线性层参数）的结果还是很不错的，如Table1所示，ResNet-50能达到74.3%的top1 Acc，这个结果甚至要优于相同网络结构的SimCLR v2的结果（71.7%）。

BYOL相比SimCLR系列的一个有趣的点在于前者对batch size和数据增强更加鲁棒，论文中也针对这2个方面做了对比实验，如Figure3所示。大batch size对于训练机器要求较高，在SimCLR系列算法中主要起到提供足够的负样本对的作用，而BYOL中没有用到负样本对，因此更加鲁棒。数据增强也是同理，对对比学习的影响比较大，因此这方面BYOL还是很有优势的。

## Why no 训练崩塌

1. online network和target network并不是由一个损失函数来共同优化，也就是target network采用了slow-moving average的方式进行参数更新，参考Algorithm中的第11行。
2. online network和target network的网络结构并不是完全一样的，online network还多了一个predictor结构。