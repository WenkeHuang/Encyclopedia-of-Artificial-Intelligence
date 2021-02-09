# Few-shot learning of neural networks from scratch by pseudo example optimization 

## Problems

所有目前的方法需要大量无监督的训练示例或通过大量监督训练数据进行训练的预训练模型，从而进行神经网络学习仅通过几个例子，仍然是一个关键挑战。

Meanwhile, several other estimators such as support vector machines (SVMs) and Gaus-sian processes (GPs) can ease the adverse effect of overfitting by making use of Bayesian principle or maximum margin.

The universal approximator theorem guarantees that an infinitely wide neural network with at least one hidden layer can represent any Lipschitz continuous function to an arbitrary degree of accuracy. This theorem implies that there exists a neural network that well imitates the behavior of other estimators, while keeping a great representation power of neural networks.（在人工神经网络的数学理论中，**通用近似定理**（或称**万能近似定理**）指出人工神经网路近似任意函数的能力。通常此定理所指的神经网路为前馈神经网路，并且被近似的目标函数通常为输入输出都在欧几里得空间的连续函数。但亦有研究将此定理扩展至其他类型的神经网路，如卷积神经网路、放射状基底函数网路、或其他特殊神经网路。

此定理意味着神经网路可以用来近似任意的复杂函数，并且可以达到任意近似精准度。但它并没有告诉我们如何选择神经网络参数（权重、神经元数量、神经层层数等等）来达到我们想近似的目标函数。）

## Idea

与以往几乎所有需要大量标记训练数据的知识蒸馏工作不同，所提出的方法仅需要少量训练数据。 相反，我们介绍了伪训练示例(pseudo training examples)，这些示例已作为模型参数的一部分进行了优化

与几乎所有以前的知识蒸馏工作都采用大量有监督的训练示例不同，我们提出的方法只需要几个有监督的训练示例即可进行知识转移。为了增强训练样本，我们引入了归纳点（inducing points），它们是伪训练示例，有助于模型训练变得容易。在用于可伸缩GP推理的原始归纳点方法中，归纳点和模型参数均被更新以增加目标函数，该函数实际上是边缘化可能性的下限（ELBO）。然而，在我们提出的方法中，目标模型的参数被更新以减少训练损失，而伪训练示例被更新以增加训练损失。通过这样做，我们可以将伪训练示例移至当前目标模型尚未得到良好训练的区域。我们还引入保真度加权(fidelity weighting)，以根据从参考模型获得的预测中的不确定性消除有害的伪训练示例。

### 相比于现有工作

**Techniques for avoiding overfitting**

不需要额外的训练数据，相比于半监督训练

从零开始即可完成few-shot学习，这意味着参考模型是通过一些训练示例进行训练的，而无需其他示例或通过大量监督训练示例进行训练的参考模型。

**Knowledge distillation**
$$
L_{dis}=(X^L,Y^L) = \frac{\lambda_1}{N_L}\sum_{n=1}^{N_L}D_1(y_n^L,f(x_n^L))+\frac{\lambda_2}{N_L}\sum_{n=1}^{N_L}D_2(g(x_n^L),f(x_n^L))\
$$
使用有限的训练样本来降低$L_{dis}$会带来过拟合的影响

imitation networks 是用pseudo traning examples $X^p = (x_1^p,...,x_{N_p}^p)$ 来训练目标模型$f(\cdot)$ 从 $g(\cdot)$ 中，并且最小化下Loss 函数：
$$
L_{imi}(X^L,Y^L,X^P) = \frac{\lambda_1}{N_L}\sum_{n=1}^{N_L}D_1(y_n^L,f(x_n^L))+\frac{\lambda_2}{N_P}\sum_{n=1}^{N_P}D_2(g(x_n^P),f(x_n^P))
$$

1. 方法继承了知识提炼的思想
2. 与标准知识蒸馏相比，利用任意的黑匣子估计器(estimator)作为教师
3. 并且不需要大量的真实训练数据，而是采用**在模型训练过程中优化的**伪训练数据。

### Step 1 Train reference model

训练一个reference model 使用有监督的样本数据(X^L,Y^L)$，但是有监督的样本是很少的在这个问题的设定中

reference model 是

### Step 2 Transfer knowledge

transfers knowledge from the reference model to a target neural network model in a similar manner to knowledge distillation



## 利普希茨连续

对于在实数集的子集的函数$f:D \subseteq \mathbb{R} \rightarrow \mathbb{R}$ ，若存在常数*K*，使得$|f(a)-f(b) \leq  K|a-b| \forall a,b \in D$ ，则称 *f* 符合利普希茨条件，对于$f$最小的常数$K$ 称为 $f$ 的利普希茨常数。 

若$K < 1$，则$f$称为收缩映射

