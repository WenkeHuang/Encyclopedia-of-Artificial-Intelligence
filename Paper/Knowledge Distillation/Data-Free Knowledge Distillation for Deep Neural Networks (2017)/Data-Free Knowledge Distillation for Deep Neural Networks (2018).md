# Data-Free Knowledge Distillation for Deep Neural Networks 

## Problems

模型压缩的提供了将大型神经网络压缩到其原始大小的一小部分，同时又保留了大多数（即使不是全部）准确性的过程。 **但是，所有这些方法都依赖于对原始训练集的访问**，如果要压缩的网络是在非常大的数据集上或在其释放引起隐私或安全问题的数据集上进行训练的，则可能并非总是可能的。 生物识别任务的案例。 我们提出了一种无数据知识蒸馏的方法，该方法能够将在大规模数据集上训练的深度神经网络压缩到其大小的一小部分，**而仅利用一些额外的元数据来提供预训练的模型发布**。 我们还将探讨可用于我们的方法的不同类型的元数据，并讨论使用每种元数据所涉及的权衡。



- Can we leverage metadata about a network to allow us to compress it effectively without access to its original training data？（“我们是否可以利用有关网络的元数据来允许我们在不访问其原始培训数据的情况下有效地对其进行压缩？” 我们提出了一种基于知识提炼的新型神经网络压缩策略，该策略利用网络在其训练集上的激活总结摘要（leverages summaries of the activations of a network on its training set）来压缩该网络而无需访问原始数据。）

## 相关工作

### 权重量化

权重量化尝试在单个神经元权重的水平上压缩网络，保留网络的所有参数，并仅尝试使用更少的空间表示每个单独的参数。 现有技术的权重量化方法即使每个参数只有两个或三个，也可以导致非常高的分类精度。

### 网络剪枝

网络修剪最初是在Lecun的“ Optimal Brain Damage”论文[13]中提出的，它试图通过将整个神经元权重完全清零来直接减少参数数量。 进一步的工作表明，这不仅是减少网络总体内存配置文件的有效方法，而且还是应对过度拟合的好方法[7]，可以帮助网络更好地推广。 最近的方法表明，有一些修剪方法可以压缩网络而不会损失任何准确性[6]。

### 知识蒸馏

代替修剪网络的权重或神经元，另一种方法称为“知识蒸馏”，它会训练较小的“学生”网络来复制较大的“老师”网络的动作。 通常可以通过尝试训练较浅的学生网络[2，3]或较薄的学生网络[19]来匹配“教师”网络的输出来完成此操作。 Hinton等人提出了一种可以从中广泛借鉴的广义方法。 [8]：它依赖于修改教师网络的最后一层，以便代替输出分类，它输出按比例缩放到某些温度参数的活动，以尝试提供有关教师模型如何概括的更多信息。

## Idea

在原始数据集上训练教师网络后，我们计算网络中每一层的激活记录，并将这些记录与模型一起保存。 这些可以采用不同的形式。

为了训练学生模型而无需访问原始数据，我们尝试仅使用教师模型及其元数据（以预先计算的激活记录的形式）来重建原始数据集。试图找到其表示与网络给出的图像最匹配的一组图像。 我们将随机的高斯噪声作为输入传递给教师模型，然后将梯度应用于该输入噪声，以最小化激活记录与噪声图像之间的差异。 重复执行此操作可以使我们部分重构教师模型对其原始训练集的看法。

Given a  neural network representation $\phi$ 

An initial network activation or proxy thereof $\phi_0 = \phi (x_0)$

Find the image $x^*$ of width $W$ and height $H$
$$
x^* = \mathop{argmin}_{x\in R^{H \ x \ W}}\ l(\phi(x),\phi_0)
$$
Where $l$ is some loss function that compares the image representation $\phi (x)$ to the target one $\phi_0$

### Top Layer Activation Statistics

我们保留的最简单的激活记录是**教师分类层中每个单元的均值和协方差矩阵**。 这也是常规知识蒸馏的策略。
$$
\mu_i = Mean(L_i/T) 
$$

$$
Ch_i = Chol(Cov(L_i/T))
$$

Where $L$ refers to the values in the network right before the final softmax activation, $i$ refers to the $i-th$ unit in that top layer, and $T=8$ referes to some temperature scaling  parameter.

为了重构输入，我们首先从这些统计数据中采样并对其应用ReLU。 然后，我们用**ReLU替换学生最上层的的非线性**，并通过优化网络输入来最小化这两个激活之间的MSE损失，从而重建一个输入以重新创建采样的激活。

### All Layers Activation Statixtics

不幸的是，上述方法的约束是不足的：有许多不同的输入可以导致相同的顶层激活，这意味着我们的重构无法将学生模型训练到非常高的精度。

**所使用的优化目标是每层MSE的总和，由该层中隐藏单元的数量标准化。** 为了确保每一层的相对重要性一致，此归一化很重要。

但是，仅凭所有层的统计数据进行重构并不能保留inter-layer dynamics of chains of neurons that specialized together to perform some computation。 为了保留这些内容，我们冻结了每批重构示例的dropout filters。 这样，某些神经元的效果将在每一层的激活中归零，从而迫使其他神经元进行补偿。可以看到，这种添加使重构集在视觉上更类似于原始数据集。 并且，无数据蒸馏后的学生准确度比没有过滤器的学生准确度稍差。

### Spectral Methods 光谱方法

为了更好地捕获网络各层之间的所有交互，我们尝试压缩教师网络的全部激活，而不是对其进行统计汇总。(compress the entirety of the teacher network’s activations rather than summarizing it statistically)

许多常用的信号压缩技术都基于**在一些正交的基础上扩展信号的思想，并假设大多数信息存储在这些基础的一小部分中**。 如果我们将神经网络的完全连接层表示为图，并将其激活表示为图信号，则可以利用[20]中提出的**图傅立叶变换（FG）**的公式来计算网络激活的稀疏基础 对于给定的类别。

$G(V,A)$ 

Where $V$ is a set of vertices corresponding to each neuron in a layer of the network. 

A is the adjacency matrix corresponding to the weighted connections of those neurons to each other.

我们可以将网络激活 $s$ 表示为实值图信号。

$s = [s_0,s_1,...s_{N-1}]^T \in R$ where each element $s_n$ is indexed by a vertex $v_n$ of the graph.

然后，我们可以通过计算其图傅立叶基础并仅保留其最大频谱系数的一小部分来压缩该图信号
$$
A: \ A = VJV^{-1} 
$$
Where $F=V^{-1}$ is the graph Fourier transform matrix, with the frequency content $\widehat{s}$ of $s$ given by $\widehat{s}=FS$.

然后，我们可以通过仅保留最大幅度的频谱系数$\widehat{s}$的一部分$C$来压缩网络的激活。 然后，通过简单地反转初始傅立叶变换矩阵并乘以频谱系数的零填充矩阵，即可完成原始信号的重建:
$$
\overline{s}=F_G^{-1}(\widehat{s}_0,...\widehat{s}_{C-1},0,...,0)^T
$$
给定一组初始的频谱系数$\widehat{s}$和相应的图傅立叶变换矩阵，我们可以计算网络的重构损耗，作为重构网络激活$\overline{s}$与$s_i$在给定迭代中的激活之间的欧几里德距离：
$$
l=\sum_i (\overline{s}-s_i)^2
$$
重构的精度取决于保留的频谱系数的数量，因此空间效率比上面使用的简单统计方法低，但是我们发现即使仅保留$10 ％$的频谱系数也可以产生很高的重构准确度。 值得注意的是，计算大型矩阵的本征分解可能会很昂贵，因此我们还考虑基于频谱的重构来重建教师网络激活，该频谱是由图对的连接给出的较小图的网络层，而不仅仅是使用整个图的层。

## 实验

选择了两个数据集来检验所提出的蒸馏方法的不同质量：MNIST它被用作所提出方法的概念验证，并提供了可以与Hinton等人直接比较的结果和CelebA，这表明我们的方法可以扩展到大型数据集和模型。

### MNIST - Fully Connected Models

**对于MNIST**，我们提取了一个完全连接的模型和一个卷积模型，以显示该方法的多功能性。 他们使用Adam训练了10个时期，学习率为0.001。 任何蒸馏程序（包括我们的方法）在重建的数据集上运行了30个周期。 首先将每个重建的输入初始化为每像素$〜N(0.15，0.1)$，然后使用Adam进行优化。

对于全连接模型的实验，我们使用了Hinton等人描述的网络。 [8]。 由两个1200个单位的隐藏层组成的网络（Hinton-784-1200-1200-10）被用作教师模型，并使用辍学进行了训练。 学生使用了另外两个由800个单位组成的隐藏层的网络（Hinton-784- 800-800-10）。 参数总数减少了50％。 对于每一个，所使用的温度参数都是8，就像Hinton一样。
首先，我们在MNIST上直接培训了教师和学生模型。 然后，我们通过使用知识传播训练学生模型来复制结果。

### MNIST - Convolutional Models
对于卷积模型的实验，我们使用LeNet-5 [14]作为教师模型，并使用每层卷积过滤器数量的一半（LeNet-5-half）的改进版本作为学生模型。 参数总数减少了约50％

### CelebA - Convolutional Models
为了使实验更接近生物统计学领域，并证明我们的方法可以推广到更大的任务，我们使用分类模型来评估我们的方法，该模型使用以下方法对大型面部属性数据集CelebA 中的最平衡属性进行分类 较大的卷积模型AlexNet 。 和以前一样，我们使用学生模型AlexNet-Half，每个卷积层的过滤器数量减半。
值得注意的是，我们发现，随着协方差矩阵以更高的速率增长，较大的卷积层的“所有层”优化目标的缩放效果很差。 表中显示了其他方法的结果

## Discussion

随着需要大量参数的深度学习方法的日益普及，考虑是否存在更好的分配学习模型的方法将非常有用。 我们假设在训练时或训练后不久可能值得收集元数据，这可能有助于压缩和分发这些模型。

然而，关于这样的元数据做出的不同选择在产生的压缩精度和存储器配置文件方面可以具有不同的权衡。 简单统计方法易于计算，几乎不需要其他参数，但是即使将配置文件的整体内存减少了50％，压缩精度也受到了限制。 与传统图像压缩策略更相似的方法要求以谱系数矢量的形式保留更多的元数据，从而产生更精确的压缩，但计算量却大大增加。

## Figures And Tables

|          Model          |               Activation Record               | Accuracy on test set |
| :---------------------: | :-------------------------------------------: | :------------------: |
| Hinton-784-1200-1200-10 |                Train on MNIST                 |        96.95         |
|  Hinton-784-800-800-10  |                Train on MNIST                 |        95.70         |
|  Hinton-784-800-800-10  |            Knowledge Distillation             |        95.74         |
|  Hinton-784-800-800-10  |             Top Layer Statistics              |        68.75         |
|  Hinton-784-800-800-10  |             All Layers Statistics             |        76.38         |
|  Hinton-784-800-800-10  | All Layers Statistics + Fixed Dropout Filters |        76.23         |
|  Hinton-784-800-800-10  |              All-Layers Spectral              |        89.41         |
|  Hinton-784-800-800-10  |             Layer-Pairs Spectral              |        91.24         |

![img_1](img_1.PNG)

|    Model     |   Activation Record    | Accuracy on test set |
| :----------: | :--------------------: | :------------------: |
|   LeNet-5    |     Train on MNIST     |        98.91         |
| LeNet-5-half |     Train on MNIST     |        98.65         |
| LeNet-5-half | Knowledge Distillation |        98.91         |
| LeNet-5-half |  Top Layer Statistics  |        77.30         |
| LeNet-5-half | All Layers Statistics  |        85.61         |
| LeNet-5-half |  All-Layers Spectral   |        90.28         |
| LeNet-5-half |  Layer-Pairs Spectral  |        92.47         |

![img_2](img_2.PNG)



|    Model     |       Procedure        | Accuracy on test set |
| :----------: | :--------------------: | :------------------: |
|   ALEXNET    |    Train on CelebA     |        80.82         |
| ALEXNET-half |    Train on CelebA     |        81.59         |
| ALEXNET-half | Knowledge Distillation |        69.53         |
| ALEXNE-half  |  Top Layer Statistics  |        54.12         |
| ALEXNET-half |  All-Layers Spectral   |        77.56         |
| ALEXNET-half |  Layer-Pairs Spectral  |        76.94         |

## Relative Rource

https://raphagl.com/research/replayed-distillation/

https://github.com/irapha/replayed_distillation

