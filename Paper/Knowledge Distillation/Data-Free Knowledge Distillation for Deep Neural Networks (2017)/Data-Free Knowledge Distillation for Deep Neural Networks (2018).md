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

### Activation Records

