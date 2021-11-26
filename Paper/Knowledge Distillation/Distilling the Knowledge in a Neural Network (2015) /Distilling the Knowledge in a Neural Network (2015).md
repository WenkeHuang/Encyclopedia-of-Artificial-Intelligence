# Distilling the Knowledge in a Neural Network（2015）

## Challenges 

**Too cumbersome to Deploy to the client**

**Overfitting**

Q:The conflicting constraints of learning and using

The easiest way to extract a lot of knowledge from the  training data is to learn many different models in parallel.

- We want to make the models as different as possible  to minimize the correlations between their errors. 

- We can use different initializations or different  architectures or different subsets of the training data. 
- It is helpful to over-fit the individual models. 

A test time we average the predictions of all the models  or of a selected subset of good models that make  different errors. 

- That’s how almost all ML competitions are won  (e.g. Netflix)

Q: Why ensembles are bad at test time

A big ensemble is highly redundant. It has very  very little knowledge per parameter. 

At test time we want to minimize the amount of computation and the memory footprint.

- These constraints are generally much more  severe at test time than during training.

## Main Idea

模型压缩和加速

“蒸馏”（distillation）：把大网络的知识压缩成小网络的一种方法 

“专用模型”（specialist models）：对于一个大网络，可以训练多个专用网络来提升大网络的模型表现

神经网络通常通过使用“softmax”输出层来产生类概率，该输出层通过将logit后的结果相互比较转化：
$$
q_i = \frac{exp(z_i/T)}{\sum_iexp(z_j/T)}
$$

1. 当温度T越高的时候，软目标越平滑，信息不会集中在少数分量上，这样增大温度参数T相当于放大（蒸馏出来）这些小概率值分量所携带的信息；
2. 不管温度T怎么取值，Soft target都有忽略小的zi携带的信息的倾向（产生的Prob小）

在常规中设置的T=1其中T是通常设置为1的温度。使用较高的T值可以在类上产生更软的概率分布。

### 迭代步骤

在最简单的蒸馏形式中，通过在迁移集上训练并将迁移集中的每个类别使用软目标分布表示，将知识转移到蒸馏的小模型，软目标分布通过使用在softmax中具有高温的繁琐模型产生。训练蒸馏模型时使用相同的高温，但训练完成后，温度设置为1。

当已知所有或部分迁移集的正确标签时，通过训练蒸馏模型以产生正确的标签，可以显著改善该方法。一种方法是使用正确的标签来修改软目标，但我们发现更好的方法是简单地**使用两个不同目标函数的加权平均值**。第一个目标函数是具有软目标的交叉熵，该交叉熵是从蒸馏模型的具有高温的softmax函数中获得的，该高温值与繁重模型生成软目标交叉熵时的温度值相同。第二个目标函数是具有正确标签的交叉熵。这是使用与蒸馏模型的softmax中完全相同的logits计算的，但温度值为1。

我们发现通常通过在第二目标函数上使用可忽略不计的较低权重来获得最佳结果。由于软目标产生的梯度的大小为$1/T^{2}$，因此当使用硬目标和软目标时，将软目标乘以$T^2$是很重要的。这确保了如果在试验元参数时改变了用于蒸馏的温度，则硬和软目标的相对贡献保持大致不变。

迁移集中的每个例子相对于蒸馏模型的每个logit  $z_i$ 贡献交叉熵梯度$dC / dz_i$。如果繁琐的模型具有logit $v_i$，其产生软目标概率$p_i$并且迁移训练在温度为$T$下完成，则该梯度由下式给出:
$$
\frac{\partial C}{\partial z_i} = \frac{1}{T}(\frac{e^{z_i/T}}{\sum_je^{z_j/T}}-\frac{e^{v_i/T}}{\sum_je^{v_j/T}})
$$
如果温度高于logits的幅度，我们可以近似：
$$
\frac{\partial C}{\partial z_i} \approx \frac{1}{T}(\frac{1+z_i/T}{N+\sum_jz_j/T}-\frac{1+v_i/T}{N+\sum_jv_j/T})
$$
如果我们现在假设对于每个迁移样本，logits均值为零：
$$
\frac{\partial C}{\partial z_i} \approx \frac{1}{NT^2}(z_i-v_i)
$$

## Experiment

### Experiment on MNIST

|     net     | layers | unit of each layer | activation | Regularization | test error |
| :---------: | :----: | :----------------: | :--------: | :------------: | :--------: |
| single net1 |   2    |        1600        |    relu    |    dropout     |     67     |
| single net2 |   2    |        800         |    relu    |       no       |    146     |

|      Net      | larget net  |  small net  | Temperature | test error |
| :-----------: | :---------: | :---------: | ----------- | ---------- |
| distilled net | single net1 | single net2 | 20          | 74         |

1. 相比使用hard targets训练的小模型，使用知识蒸馏训练的小模型性能更好。这说明知识蒸馏可以将大量的知识迁移到小模型中，这些知识可以提高模型的泛化能力。 
2. 作者还对net1的神经元数量做了调整。可以发现，当大模型每层神经元的数量较多时，需要的蒸馏温度更高；当每层的神经元数量较少时，需要的蒸馏温度相对较低。 
3. 作者还尝试在蒸馏训练中略去所有的数字“3”，也就是说，小模型没有学过数字“3”的特征。使用这种方式训练好的小模型预测错误206个例子，其中数字“3”预测错133个，而数字“3”总共有1010个。这说明小模型通过知识蒸馏，从其他类型的样本中学到了数字“3”的特征。作者还发现，如果提高模型对某种类型样本的偏倚，模型对该类型样本的学习能力也会有所提高。

### Experiments on speech recognition

|         System         | Test Frame Accuracy(%) | WER(%) |
| :--------------------: | :--------------------: | :----: |
|        Baseline        |          58.9          |  10.9  |
|      10xEnsemble       |          61.1          |  10.7  |
| Distilled Single model |          60.8          |  10.7  |

Basline的配置为 8 层，每层2560个relu单元 softmax层的单元数为14000 训练样本大小约为 700M，2000个小时的语音文本数据

10XEnsemble是对baseline训练10次（随机初始化为不同参数）然后取平均

蒸馏模型的配置为 使用的候选温度为{1, 2, 5, 10}, 其中T为2时表现最好 hard target 的目标函数给予0.5的相对权重 可以看到，相对于10次集成后的模型表现提升，蒸馏保留了超过80%的效果提升

### Training ensembles of specialists on very big datasets

训练一个大的集成模型可以利用并行计算来训练，训练完成后把大模型蒸馏成小模型，但是另一个问题就是，训练本身就要花费大量的时间，这一节介绍的是如何学习专用模型集合，集**合中的每个模型集中于不同的容易混淆的子类集合**，这样可以减小计算需求。专用模型的主要问题是容易集中于区分细粒度特征而导致过拟合，可以使用软目标来防止过拟合。

JFT数据集：JFT是一个谷歌的内部数据集，有1亿的图像，15000个标签。google用一个深度卷积神经网络，训练了将近6个月。我们需要更快的方法来提升baseline模型。

专用模型：将一个复杂模型分为两部分，**一部分是一个用于训练所有数据的通用模型，另一部分是很多个专用模型，每个专用模型训练的数据集是一个容易混淆的子类集合。**这些专用模型的softmax结合所有不关心的类为一类来使模型更小。为了减少过拟合，共享学习到的低水平特征，每个专用模型用通用模型的权重进行初始化。另外，专用模型的训练样本一半来自专用子类集合，另一半从剩余训练集中随机抽取

将子类分配到专用模型：专用模型的子类分组**集中于容易混淆的那些类别**，虽然计算出了混淆矩阵来寻找聚类，但是可以使用一种更简单的办法，不需要使用真实标签来构建聚类。对通用模型的预测结果计算协方差，根据协方差把经常一起预测的类作为其中一个专用模型的要预测的类别。几个简单的例子如下。

> JFT 1: Tea party; Easter; Bridal shower; Baby shower; Easter Bunny; ...
>
> JFT 2: Bridge; Cable-stayed bridge; Suspension bridge; Viaduct; Chimney; ...
>
> JFT 3: Toyota Corolla E100; Opel Signum; Opel Astra; Mazda Familia; ...

|        System         | Conditional Test Accuracy(%) | Test Accuracy |
| :-------------------: | :--------------------------: | :-----------: |
|       Baseline        |             43.1             |     25.0      |
| +61 Specialist models |             45.9             |     26.1      |

## Aside

### Softmax求导

[简单易懂的softmax交叉熵损失函数求导](https://blog.csdn.net/qian99/article/details/78046329)

### Two ways to average models

We can combine models  by averaging their  output probabilities:

|          | class 1 | class 2 | class 3 |
| :------: | :-----: | :-----: | :-----: |
| Model A  |   0.3   |   0.2   |   0.5   |
| Model B  |   0.1   |   0.8   |   0.1   |
| Combined |   0.2   |   0.5   |   0.3   |

We can combine models by taking the geometric  means of their output probabilities:

|          |   class 1    |   class 2    |   class 3    |
| :------: | :----------: | :----------: | :----------: |
| Model A  |     0.3      |     0.2      |     0.5      |
| Model B  |     0.1      |     0.8      |     0.1      |
| Combined | $\sqrt{.03}$ | $\sqrt{.16}$ | $\sqrt{.05}$ |

