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

reference model$g(\cdot)$可以是一个estimator 或者一组estimators.(我们可以通过更改超参数，例如GP的RBF内核中的方差和长度比例，从单个模型构建多个estimators)

同时对于多个estimators使用他们的平均预测结果作为最终结果

### Step 2 Transfer knowledge

transfers knowledge from the reference model to a target neural network model in a similar manner to knowledge distillation

上面的表述表明目标模型试图模仿参考模型的预测，同时，它试图尽可能准确地返回受监督样本的预测能力。 因此，生成的目标模型不是参考模型的副本，并且可以继承来自参考模型和神经网络的两个不同属性。 特别是使用高斯过程作为参考模型时，**他们将预测的局部平滑度引入到神经网络中**，这是标准神经网络所缺乏的，并被证明对于过拟合有效。

同时从参考模型得出不可靠预测的示例更可能有害，因此应在模型训练中丢弃。

**Fidelity weighting**
$$
L_{imi}(X^L,Y^L,X^P) = \frac{\lambda_1}{N_L}\sum_{n=1}^{N_L}D_1(y_n^L,f(x_n^L))+\frac{1}{N_P}\sum_{n=1}^{N_P}\lambda_2(g,x_n^P) D_2(g(x_n^P),f(x_n^P))
$$
这里 $\lambda_2(g,x_n^P)$是一个新的样本期待的权重对于$x_n^P$，基于不确定的$\sigma_g(x_n^P)$ 来自参考模型$g(x_n^P)$的预测结果。
$$
\lambda_2(g,x_n^P) = \widehat{\lambda}_2 exp (-log(\widehat{\lambda}_2/\overline{\lambda}_2)\sigma_g(x_n^p)/\overline{\sigma}_g)
$$
$\widehat{\lambda}_2$ 代表的权重的上限取值

$\overline{\sigma}_g$ 代表的平均不确定性基于所有的pseudo样本

$\overline{\lambda}_2$ 是平平均不确定性对应的权重

## Pseudo example 优化

The inducing point method has been originally developed for scalable GP inference 在这种方法中，归纳点（inducing points）和模型参数都被更新以更新目标函数。

模型参数更新时为了降低imitation的损失

pseudo 例子更新时为了增加这一损失

通过这样做，我们可以将伪示例移至当前目标模型尚未经过良好训练的区域。

提出的用于更新伪示例的技术的灵感来自对抗训练，这增加了神经网络对对抗示例的鲁棒性，这是通过对数据集中的样本的较小但有意的最坏情况的摄动而形成的。
$$
x_{AT}(x^L,y^L) = x^L + \varepsilon \ sign \{\nabla_{x^L}D_1(y^L,f(x^L))\}
$$
$\nabla_x$ 是x的partial derivative，同时$\varepsilon$是一个小常数且$\varepsilon \geq 0$

对抗性样本$x_{AT}(x^L,y^L)$可以看作是原始样例$x$的随机更新，从而增加了损失，并且如果梯度符号的部分符号为$sign \{\nabla_{x^L}D_1(y^L,f(x^L))\}$由标准随机梯度$\nabla_{x^L}D_1(y^L,f(x^L))$代替。

同样更新 pseudo 训练样本
$$
x_{imi}(x^P) = x^P + \varepsilon \nabla_{x^P} D_2(g(x^P),f(x^P))
$$
然而，众所周知，这种对抗训练过程是不稳定的[29]。 取而代之的是，我们采用了另一种增加伪示例的方法。就是固定每个周期的训练样本的数据 不改动 然后训练完这一周期 再统一更新。

## 实验

### Banana 数据集

为了定性地检查所提出方法的行为，我们首先将Banana数据集进行了400个二维示例的二进制分类。 我们通过线性变换将两个二维示例分别嵌入到100维向量空间中，在该空间中预先确定了一个转换矩阵，并且我们每个类别随机选择了五个监督训练示例。

所选的受监管示例和其他未使用示例显示在图中，其中十字是受监管示例，点是数据集中的未使用示例，并且颜色对应于其类别标签。

我们使用带有RBF内核的GP分类器作为参考模型，并建立了多个使用不同初始内核参数训练的参考模型的集合。（We used GPflow with variational Gaussian approximation and L-BFGS-B optimization for GP inference, and utilized mean predictions as pseudo supervisions.）

我们选择的目标模型是7层全连接神经网络，其中每个中间层具有1000个单位。 我们使用Nadam 进行模型参数优化，使用Adam 进行伪示例优化，它们的初始学习率分别设置为0.001和0.05。 批次大小为100，训练时期为200。

![Experiment_1](Experiment_1.PNG)

orange & blue crosses = supervised examples （有监督样本）

orange & blue dots = other unused examples in the dataset （其他没使用的样本）

green dots = pseudo examples（伪样本）

black lines = classification boundaries （分类边界）

a) a distribution of examples 代表数据的分部

b) densely distributed pseudo examples 代表密集分布的伪样本数据

c) a reference model trained with supervised examples 参考模型使用有监督数据训练的结果

d) a target model trained with densely distributed pseudo examples 目标模型使用密集的伪样本数据

 e) pseudo examples generated from supervised examples as seeds 使用有监督样本作为seed的数据 随机生成一批

f) a target model trained with generated pseudo examples  更新模型使用生成的伪生成数据

g) a target model trained with supervised and pseudo examples.  使用有监督和随机的伪样本的数据更新的模型

h) a target model trained with supervised and pseudo examples + fidelity weighting 考虑了fidelity 权重的更新

首先，我们研究了极端情况，其中2500个伪示例密集分布在整个特征空间上。 尽管这对于高维空间尤其不现实，但检查我们提出的方法是否可以精确模拟参考模型的行为非常有用。 b）显示了它的分布，其中绿点是伪示例。 在这种情况下，我们没有优化伪示例，也没有采用保真度加权，这意味着我们利用了等式中所示的原始模仿损失。 我们将Kullback-Leibler发散用于软损耗D2，而忽略了硬损耗D1（即λ1= 0）。 c）和d）分别显示了参考模型和目标模型的决策边界。 结果表明，我们的方法几乎完全像预期的那样模仿了参考模型的预测。

### MNIST和Fashion-MNIST

接下来，我们对几种基准数据集定量评估了该方法的分类性能。 我们使用MNIST [21]和时尚MNIST [36]作为该实验的数据集。 我们再次采用GP分类器作为参考模型，其中与参考模型相关的所有设置均与第5.1节相同。 目标模型是两个数据集的3层CNN，可以在补充材料中找到详细的配置。 我们再次将Nadam用于模型参数优化，将Adam用于伪示例优化，并且两个初始学习率均设置为0.02。 我们通过插值两个不同的受监管示例准备了1.25K个初始伪示例，并通过使用第4.1节中所示的技术将其扩展为10K。 因此，整个训练过程包含8个训练步骤，每个训练步骤都有25个训练时期。

## 总结

在本文中，提出了一种简单但有效的方法，用于训练神经网络，但训练示例数量有限。 提出的方法以GP为参考，并建立了目标神经网络，以模仿受限实例训练的参考行为。 我们介绍了伪示例，并通过目标模型训练过程对其进行了优化。 由于提出的建议框架是通用的，因此可以直接应用于参考模型和目标模型的其他组合。 例如，可以用在少量数据上训练的预训练浅参考网络训练深目标网络，这是知识提炼的逆过程，这可以提供预训练深度神经网络的新途径。 同时，我们用于优化伪示例的方法相当具体，并且伪示例的复杂管理仍需大量研究。

## 利普希茨连续

对于在实数集的子集的函数$f:D \subseteq \mathbb{R} \rightarrow \mathbb{R}$ ，若存在常数*K*，使得$|f(a)-f(b) \leq  K|a-b| \forall a,b \in D$ ，则称 *f* 符合利普希茨条件，对于$f$最小的常数$K$ 称为 $f$ 的利普希茨常数。 

若$K < 1$，则$f$称为收缩映射

