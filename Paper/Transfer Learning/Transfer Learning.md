# 迁移学习是个啥

## 引子

 举个例子，譬如三个月后 武汉会变得很热，武汉如此，北京，纽约都很热。然而此刻，假如我问你，阿根廷的首都布宜诺斯艾利斯，天气如何？稍稍有点地理 常识的人就该知道，阿根廷位于南半球，天气恰恰相反：正是夏末秋初的时候，天气渐渐凉了起来。

我们何以根据北京的天气来推测出纽约、东京和巴黎的天气？我们又何以不能用相同 的方式来推测阿根廷的天气？

因为它们的地理位置不同。除去阿根廷在南半球之外，其他几个城市均位于北半球，故而天气变化相似。

因而根据地点的相似性和差异性，很容易地推测出其他地点的天气。 这样一个简单的事实，就引出了我们要介绍的主题：**迁移学习**

## 概念

迁移学习作为机器学习的一个重要分支，侧重于将已经学习过的知识迁移应用于新的问题中

**找到新问题和原问题之间的相似性**

Example：

举一反三，照猫画虎，数学&物理

## 为啥需要

1. **大数据与少标注之间的矛盾**

<img src="./img/Lecun_Cake.png" alt="Lecun_Cake" style="zoom:80%;" />

大数据带来了严重的问题：总是缺乏完善的数据标注。

机器学习模型的训练和更新，均依赖于数据的标注。然而，尽管我们可以获 取到海量的数据，这些数据往往是很初级的原始形态，很少有数据被加以正确的人工标注。

2. **大数据与弱计算之间的矛盾**

大数据，就需要大设备、强计算能力的设备来进行存储和计算。

我电脑就一张显卡，还时不时被我跑到宕机

3. **普适化模型与个性化需求之间的矛盾。**

对于每一个通用的任务都构建了一个通用的模型。这个模型 可以解决绝大多数的公共问题。但是具体到每个个体、每个需求，都存在其唯一性和特异 性，一个普适化的通用模型根本无法满足。那么，能否将这个通用的模型加以改造和适配， 使其更好地服务于人们的个性化需求？

4. **特定应用的需求**

机器学习已经被广泛应用于现实生活中。在这些应用中，也存在着一些特定的应用，它 们面临着一些现实存在的问题。比如推荐系统的冷启动问题。一个新的推荐系统，没有足够的用户数据，如何进行精准的推荐? 一个崭新的图片标注系统，没有足够的标签，如何进行 精准的服务？现实世界中的应用驱动着我们去开发更加便捷更加高效的机器学习方法来加 以解决。

| 矛盾                   | 传统机器学习                     | 迁移学习                                        |
| ---------------------- | -------------------------------- | ----------------------------------------------- |
| 大数据与少标注         | 增加人工标注，但是昂贵且耗时     | 数据的迁移标注(Domain Adaptation)               |
| 大数据与弱计算         | 只能依赖强大计算能力，但是受众少 | 模型迁移(Knowledge Distillation)                |
| 普适化模型与个性化需求 | 通用模型无法满足个性化需求       | 模型自适应调整 (Personalized Federaed Learning) |
| 特定应用               | 冷启动问题无法解决               | 数据迁移 (Self-Supervised Learning)             |

## 分类

迁移学习的分类可以按照四个准则进行：按目标域有无标签分、按学习方法分、按特征分、按离线与在线形式分

**按目标域标签分** 

这种分类方式最为直观。类比机器学习，按照目标领域有无标签，迁移学习可以分为以 下三个大类： 

1. 监督迁移学习 (Supervised Transfer Learning) 
2. 半监督迁移学习 (Semi-Supervised Transfer Learning) 
3. 无监督迁移学习 (Unsupervised Transfer Learning)

**按学习方法分类**

1. 基于样本的迁移学习方法 (Instance based Transfer Learning) 

   通过权重重用，对源域和目标域的样例进行迁移。就是 说直接对不同的样本赋予不同权重，比如说相似的样本，我就给它高权重，这样我就完成了 迁移，非常简单非常非常直接。

2.  基于特征的迁移学习方法 (Feature based Transfer Learning) 

   假设源域和目标域的特征 原来不在一个空间，或者说它们在原来那个空间上不相似，那我们就想办法把它们变换到 一个空间里面

3.  基于模型的迁移学习方法 (Model based Transfer Learning) 

   构建参数共享的模型。这个主要就是在神经网络里面用的特 别多，因为神经网络的结构可以直接进行迁移。比如说神经网络最经典的 finetune 就是模 型参数迁移的很好的体现

4. 基于关系的迁移学习方法 (Relation based Transfer Learning)

   挖掘和利用关系进行类比迁 移。比如老师上课、学生听课就可以类比为公司开会的场景。这个就是一种关系的迁移。

**按特征分类**

1. 同构迁移学习 (Homogeneous Transfer Learning)  

2. 异构迁移学习 (Heterogeneous Transfer Learning)

这也是一种很直观的方式：如果特征语义和维度都相同，那么就是同构；反之，如果特 征完全不相同，那么就是异构。举个例子来说，不同图片的迁移，就可以认为是同构；而图 片到文本的迁移，则是异构的。

**按离线与在线形式分**

1. 离线迁移学习 (Offline Transfer Learning) 
2. 在线迁移学习 (Online Transfer Learning)

#大数据与少标注 Domain Adaptation

## Unsupervised Domain Adaptation by Backpropagation

在MNIST数据集上训练出来的网络，同样去跑相近的数据集，比如MINIST-M，准确率大打折扣？或者是想跑另外一个没有标签的数据集却没有精力去标注？

对于一个常规的网络而言，一般由特征提取backbone和分类器classifer组成

<img src="./img/UDA.jpeg" alt="UDA" style="zoom:80%;" />

source domain的数据是有标签的

target domain的数据是无标签的。

$G_f$ 将source和target domain的数据都映射到一个特征空间$Z$上 

$G_y$预测标签y

$G_d$ 预测数据来自于target还是source domain

流入$G_y$的是带标签的source数据

流入$G_d$的是不带标签的source和target的数据。

当我们不考虑七里八里的 正常的一个分类模型
$$
\theta_f \leftarrow \theta_f - \mu(\frac{\delta L_y}{\delta \theta_f})
$$

$$
\theta_y \leftarrow \theta_y - \mu(\frac{\delta L_y}{\delta \theta_y})
$$
模型要应用到数据集分为两部分：源域（source domain）和目标域（target domain）。前者是可以很方便地训练的数据集，后者是很少有标签或者就没标签的数据集。

这就是一个半监督或者无监督的学习问题。由于存在dataset bias，这个判别模型不能直接移植到目标域，否则效果会很差，在本文结构中CNN的作用主要是找到合适的features，这些features可以送到全连接分类器进行分类，同时送到domain classifier ，让domain分类器得到如何分辨domian的梯度，经过梯度反转，使得CNN搞出来的features更加迷惑domain classifier，最后使得domain classifier无法分辨出这两个domain。这时，任务就达到了，因为在无法分辨两个分类器的情况下，这些features彩色这两个domain共享的，这时把CNN网络迁移到另外这个分布，效果会好很多。



对于$\theta_f$ 他是要在确保提取有效特征的情况下，把Domain Identifier 给弄迷糊
$$
\theta_f \leftarrow \theta_f - \mu(\frac{\delta L_y}{\delta \theta_f})
$$

$$
\theta_f \leftarrow \theta_f + \mu(\frac{\delta L_d}{\delta \theta_f})
$$

$$
\theta_f \leftarrow \theta_f - \mu(\frac{\delta L_y}{\delta \theta_f}-\frac{\delta L_d}{\delta \theta_f})
$$
对于$\theta_d$ 他是要尽可能去分出这两个类
$$
\theta_d \leftarrow \theta_d - \mu(\frac{\delta L_d}{\delta \theta_d})
$$
<img src="./img/UDA_Visualization.jpeg" alt="UDA_Visualization" style="zoom:80%;" />

那么在这篇文章中，他是希望能够学习一个通用的特征提取，能够有效地提取不带Domain Gap的数据，因而是target映射到source ,然后用source的分类器来处理，那么能不能就是说希望能够不带这么严格的约束，我们希望源域和目标域能各自提取特征的方式。

## Adversarial discriminative domain adaptation

1. 先通过source domain的样本和标签训练Ms和C 

<img src="./img/ADDA/Loss_1.png" alt="Loss_1"/>

2. 保持Ms和C不变，用Ms初始化Mt，并交替优化第二项第三项，获得D和Mt 

<img src="./img/ADDA/Loss_2.png" alt="Loss_2" />

3. testing stage：target domain的sample直接通过Mt和C获得预测的标签。

<img src="./img/ADDA/Loss_3.png" alt="Loss_3"/>

![ADDA](./img/ADDA.png)

#大数据与弱计算  Knowledge Distilliation

没钱搞数据，没钱搞GPU 知识蒸馏闪亮登场

## Distilling the Knowledge in a Neural Network

将一个复杂模型的knowledge ，transfer到一个简单的模型，这也正是**蒸馏**所做的事情。

**Main Idea**

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

那么之后我们可以通过有限的数据，通过老师模型给予的SoftLabel来教授学生模型

![kd](./img/KD.png)

## Ask, acquire, and attack: data-free UAP generation using class impressions

如果 我是说如果，这个有钱训练的厂家连他们的数据也不愿意给你，虽然过于恶心，但是也是可能的，他们只把他们的模型给你，那么这种情况下该如何学习呢？

他不给？行，那咱就自己去造出来

数据没有，我们就制造标签，通过标签反向约束

![AAA](./img/AAA.png)

更新噪声直到它能使分类器输出一个指定的类，下面的图就是更新后的噪声的样子，根据这图看出一些类:
$$
CI_c = \mathop{argmax}_x f^{ps/m}_c(x)
$$


![reuslt](./img/AAA/result.png)

当然这么得到的图片有几个问题

1. 图片的丰富性不足，每个类内的差别都较小 [1,0,0,0,0] 
2. 没有考虑类之间的联系：猫和狗比猫和猪更相似

**Zero-shot knowledge distillation in deep networks (ICML 2019)** 类内关系 狄利克雷分布

**Data-free learning of student networks (ICCV 2019)** 生成器更加多样 

#普适化模型与个性化需求  Personalized Model

个人感觉 普适化与个性化产生的原因是由于各自的数据集的领域是存在着差异的，那么我们希望一个generalization的模型可以迅速适应我们的模型条件，这样的方式是符合我们的预期的。即首先应该考虑训练出一个泛化能力足够好的元模型来，而不是一个针对到具体personal的情况的模型。

先讲过有意思的例子哈哈 可能不太恰当：

如果有来生， 要做一棵树， 站成永恒， 没有悲欢的姿势。 一半在土里安详， 一半在风里飞扬， 一半洒落阴凉， 一半沐浴阳光， 非常沉默非常骄傲， 从不依靠 从不寻找。三毛《说给自己听》。好，问题来了，你的那棵树要适应一半在土里，一半在风里，一半阴凉，一半阳光，那么你自然要对这棵树进行个性化调整是吧，不能把在阴凉处的树直接搬到阳光下吧，这不合理。我们可以看到个性化是很重要的。

这个问题我们可以放到联邦学习的场景中来看：

联邦学习（Federated Learning）。该技术实际上是一种加密的分布式机器学习技术，各个参与方可在不披露底层数据和其加密形态的前提下共建模型。

人话就是：大家不愿意泄漏自己的隐私数据，但是又想提升预测能力，这在金融和医疗行业是很常见的，用户的投资数据和病人的分析数据都是高度保密的，而且也是不能被泄漏的。

但是问题来了，一个医院在美国和在中国如果想一起共同建模的话，无论咋说美国人和中国人的身体素质还是有差异的，因而如果简单的不考虑个性化的需求，可能会有很差的效果，这里先介绍联邦学习的基石：FedAVG

## Communication-Efficient Learning of Deep Networks from Decentralized Data

单独而言：
$$
\min_{\boldsymbol{x} \in \mathbb{R}^d} \quad \left[ F(\boldsymbol{x}) = \frac{1}{n} \sum_{i=1}^{n} f(\boldsymbol{x};s_i) \right],
$$
FedAvg算法的思想很直观，将训练过程分为多个回合，每个回合中选择$CK(0 \le C \le 1)$个局部模型对数据进行学习。第$k$个局部模型在一个回合中的epoch数量为$E$，batch大小为$B$，从而迭代次数为$E_{n_k}/B$。在一个回合结束之后，对所有参与学习的局部模型的参数进行加权平均得到全局模型。详细的算法如下。

![fedavg](./img/fedavg.png)

对一般的非凸目标函数来说，将多个局部模型的参数平均之后可能产生表现很差的全局模型。但是，若对多个模型进行**相同的初始化**(算法流程第7行)，在分别训练之后取参数的加权平均得到全局模型，该全局模型的表现会出乎意料地好。FedAvg算法的有效性也是来自于这个经验法则。

对一般的非凸目标函数来说，将多个局部模型的参数平均之后可能产生表现很差的全局模型。但是，若对多个模型进行**相同的初始化**(算法流程第7行)，在分别训练之后取参数的加权平均得到全局模型，该全局模型的表现会出乎意料地好。FedAvg算法的有效性也是来自于这个经验法则。

## Personalized Federated Learning with Moreau Envelopes 

但是由于不同的参与方的数据存在较大的差异的话，可能这个模型在实际各自的测试集表现是拉垮的。F限制了全局模型在每个客户的任务上提供良好的性能。
$$
\mathop{min}_{w\in R^d}\{F(w):=\frac{1}{N}\sum_{i=1}^NF_i(w)\} 
\\
where \ F_i(w) = \mathop{min}_{\theta_i \in R^d} \{f_i(\theta_i)+\frac{\lambda}{2}||\theta_i-w||^2\}
$$
通过使用Moreau Envelopes，调节Personalized和Global模型的差异程度，用lambda 控制

引入 Moreau Envelopes 作为客户机正则化损失函数的个性化联邦学习算法（pFedMe），该算法有助于将个性化模型优化与全局模型学习分离开来

虽然很大$\lambda$可以从丰富的数据聚合中受益于数据不可靠的客户，规模小$\lambda$ 帮助拥有足够有用数据的客户确定优先级，即少学一点

#特定应用 Self-Supervised Learning 

迁移学习很重要的一种形式，是把别人学到的迁移到自己的这里来，但是如果别人的数据是没有标签的呢，那么很自然，如果源数据是无标签的，我们很难说对源数据进行有监督的学习。那么自监督学习在这里会发挥很重要的作用。自监督学习通过一定的手段先训练好一个Backbone之后，那么当他连接一个MLP后，在下游的譬如分类任务的时候，可以通过冷冻住Backbone 然后训练MLP 使用少量的数据即可完成在下游任务的工作。同时这么做可以和有监督使用大量数据效果基本持平甚至超过。

<img src="./img/SSL_Development.jpg" alt="SSL_Development.jpg" style="zoom:80%;" />

在这里 着重讲两篇文章

首先，一般而言 对于一个分类网络，是由backbone和classifier两层组成的，分别负责提取特征，再根据特征去分类，那么理论上而言，我们如果能够获得一个能够有效对图像特征进行分类聚合的backbone就是可以的，那么我们便考虑如何在不使用label的情况下，训练得到了一个可以有效提取特征的backbone

##Unsupervised embedding learning via invariant and spreading instance feature

**同类别样本距离相近，从而会集中在一起；不同类别样本距离较大，从而会分散分布**	

我们的目标是在未标定的图片集$X=x_1,x_2,...,x_n$上，学习一个特征embedding 网络结构$f_{\theta}(\cdot)$。$f_{\theta}(\cdot)$经输入的图片$x_i$映射到一个低纬度的embedding特征$f_{\theta}(x_i)\in R^d$, 其中d为特征的维度。简单地，一个图片示例的特征$f_{\theta}(x_i)$记作$f_i$, 且我们假设所有特征都是L2归一化，即$||f_i||_2=1$：

我们将任意一张图片通过backbone 都会得到一个特征层embedding layer

1）视觉相似图片对应的特征embedding应该相近；

2）不相似图片对应过的特征embedding应该分散。

没有类别标签，我们使用实例监督来近似“正集中负分散”的特性。特别地，同一个实例使用不同数据增强方式得到的embedding特征应该是一致的，而不同实例应该分散。

具体而言：

典型的CNN将每张图片当做一个独立的类。依照常规的分类训练，会定义一个矩阵$W =[ w_1 , w_2 , ⋅ ⋅ ⋅ , w_n ]^T \in R^{n \times d}$.

其中$w_j$表示第j个实例对应的类别权重。经典的CNN会通过学习权重将不同图片变换的图片实例正确的分类到原始实例上。基于Softmax函数，$x_j$被识别为第i类实例的概率表达为式
$$
P(i|x_j)=\frac{exp(w_i^T f_j)}{\sum_{k=1}^n exp(w^T_kf_j)}
$$
每一步网络都会将样本特征fi拉向其对应的权重$w_i$ ，且将它从其他实例的类别权重$w_k$, 推离。然而，类别权重明显的阻碍了特征的比较，这导致效率和可分辨性受到限制。



那么核心问题出现了 如何通过Embedding的特征有效地对不同的输入进行特征的拉扯和聚合

好，假设你去韩国整容了，譬如你做了双眼皮，瘦脸，隆鼻，下巴，你做完了你本质上还是你这个没错吧，只是更符合网红的气息了，这种操作实际上是一种数据增强的操作。你整容前和现在的图片自然而言就构成了一个人的有差异的表现，但是你们还是一个人，那么是不是我还是把你认为是一个人，而不是两个人。这样子就构成了一组正样本，因为你们都是基于一个实例/样本构造的。

![Data_Argumentation](./img/Ye/Data_Argumentation.png)

是吧，你看这个鸟经过放大，或者加噪音，或者旋转切割 你还是能把她认出来。

进一步，在标题中有说Invariant and spreading instance feature。

同一样本经过数据增强 自然而然构成了实例特征的不变性，而对于第二个元素呢，这个是存在于实例之间的。

两个双胞胎长得再像那也是不同的，毕竟身份证都不一样，抑或是你按照网红整容，那不同医生动刀子还是有差别，自然而然我们希望拉开实例之间的差别，拉近实例内的差别。

为达到数据增强一致性和不同实例之间的分散性，我们打算考虑两种因素：

1）原始图片及其增强图片

2）一小批随机选择的样本而非所有数据集数据。

对于每一步迭代，我们从数据集中随机抽样m个样本。为简化标记且不失一般性，$\{x_1, x_2, · · · , x_m\}$

The augmented sample $T(x_i)$ is denoted by $\widehat{x}_i$, and its embedding feature $f_{\theta}(\widehat{x}_i)$ is denoted by $\widehat{f}_i$
$$
P(i|\widehat{x}_i)=\frac{exp(f_i^T\widehat{f}_i/\tau)}{\sum_{k=1}^mexp(f_k^T\widehat{f}_i/\tau)}
$$

$$
P(i|x_j)=\frac{exp(f_i^T f_j/\tau)}{\sum_{k=1}^mexp(f_k^T f_j/\tau)} j \neq i
$$

假设每个实例被识别为实例i是相互独立的，那么$\widehat{x}_i$被识别为实例i和$x_j (j \neq i）$不被识别为实例i的联合概率
$$
P_i=(Pi|\widehat{x}_i)\prod_{j\neq i}(1-P(i|x_j))
$$


通过极小化一个batch内的所有实例的负对数似然函数之和，来求解这个问题
$$
J = -\sum_i log P(i|\widehat{x}_i)-\sum_i\sum_{j\neq i}log(1-P(i|x_j))
$$
通过孪生网络：每一次迭代，m个随机选出的图片实例喂到第一个分支，对应的增强样本喂到第二个分支。值得注意的是，数据增强也会在第一个分支使用，来丰富训练样本。这个实现上，每个样本都有一个随机增强的正样本和2N-2个负样本类

![framework](./img/YE/framework.png)

## Barlow Twins: Self-Supervised Learning via Redundancy Reduction

相比于之前的文章，强调负样本的作用，同时很多之后的工作都要求batch_size 很大，一定要有足够多的负样本，直接把batch size搞到4096，一力降十会

那么能不能有其他的思路 我不需要这么多的负样本，batch太大 本身也是一种折磨

那么 我们能不能不从样本的角度出发，而是从embedding的角度出发？

在不考虑数据增强这种大家都有的trick的基础上，我们可以考虑embedding。

我们的目标是希望不同视角下的特征的相关矩阵接近恒等矩阵，这就意味着约束模型让不同增强的后的同一实例能够embedding足够相似，即意味着embedding 特征层两两相乘构造的矩阵，应该对角线上的元素要权重够大，其他部分的如果是0就完美了

![Barlow](./img/Barlow.png)

将样本经过不同增广送入同一网络中得到两种表示，利用损失函数迫使它们的**互相关矩阵接近于恒等矩阵**：这意味着同一样本不同的增广版本下**提取出的特征表示非常类似**，同时**特征向量分量间的冗余最小化**

损失函数如下所示，C为互相关矩阵，其中invariance term使得对角元素接近于1，促使同一样本在不同失真版本下的特征一致性，redundancy reduction term使非对角元素接近0，解耦特征表示的不同向量分量
$$
C_{ij} \triangleq \frac{\sum_b z^A_{b,i}z^B_{b,j}}{\sqrt{\sum_b (z^A_{b,i})^2},\sqrt{\sum_b z^B_{b,j}}}
$$
b表示同一batch内的样本索引，i与j表示特征的维度索引，输出值在[-1, 1]之间

损失函数如下所示，C为互相关矩阵，其中invariance term使得对角元素接近于1，促使同一样本在不同失真版本下的特征一致性，redundancy reduction term使非对角元素接近0，解耦特征表示的不同向量分量
$$
L_{BT}\triangleq = \sum_{i}(1-C_{ii})^2+\lambda\sum_i\sum_{j\neq i}C_{ij}^2
$$
众所周知，就像龙族 路明非每次拯救世界的时候，就要支付四分之一的生命去交换某些东西，something for nothing，我们假设他不知道可以不劳而获哈

你既然没有负样本了，那么希望embedding能够携带足够的特征，那么自然而然要扩大对应的embedding的大小，8192

从最早的要求大量的负样本，到通过停止梯度传播验证负样本并非不可或缺，到本文切换了一直以来的对比学习去训练的思路，转而对特征本身出发，推开了另一扇大门。对比而言，相比于最早的方法，本文仅仅是换了一个loss function (+大维度的特征)。不过，大的维度相比于增加batchsize的代价要小得多，就是多占一点的显存。

# 迁移学习的理论保障





# 引用

## 笔记

[迁移学习简明手册](http://jd92.wang/assets/files/transfer_learning_tutorial_wjd.pdf)

[《迁移学习》: 领域自适应(Domain Adaptation)的理论分析 - 小蚂蚁曹凯的文章 - 知乎](https://zhuanlan.zhihu.com/p/50710267)

### Domain Adaptation

[Adversarial Discriminative Domain Adaptation](https://zhuanlan.zhihu.com/p/67065333)

[Adversarial Discriminative Domain Adaptation阅读笔记](https://zhuanlan.zhihu.com/p/275694799)

### Knowledge Distilliation

[简单易懂的softmax交叉熵损失函数求导](https://blog.csdn.net/qian99/article/details/78046329)

### Self-Supervised

[最简单的self-supervised方法](https://zhuanlan.zhihu.com/p/355523266)

[Self-Supervised 总结](https://zhuanlan.zhihu.com/p/357830995)

[Moco三部曲](https://mp.weixin.qq.com/s?__biz=Mzg4MjQ1NzI0NA==&mid=2247494571&idx=1&sn=4b35d19caf5015b33fc40ce8a8433869&chksm=cf54c458f8234d4e6563ba94ba3a8f08c74819556a223338413e3f1c27c730915abf7842e98a&scene=21#wechat_redirect)

[Self-Supervised: 如何避免退化解](https://mp.weixin.qq.com/s?__biz=Mzg4MjQ1NzI0NA==&mid=2247494486&idx=1&sn=dd8700c12d394bcc4eb032ed09b508b7&chksm=cf54c4a5f8234db3df5d82cc7839a6d8dc38991e495bfbda0f3b75a643c1686f1fe5c93502a3&scene=21#wechat_redirect)

## 论文

### Domain Adaptation

ICML_2015	[Unsupervised domain adaptation by backpropagation](http://proceedings.mlr.press/v37/ganin15.pdf)

CVPR_2017	[Adversarial Discriminative Domain Adaptation](https://ieeexplore.ieee.org/document/8099799)

### Knowledge Distilliation

NIPS_2014 [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)

ECCV_2018 [Ask, acquire, and attack: data-free UAP generation using class impressions](https://arxiv.org/abs/1808.01153)

ICML_2019 [Zero-shot knowledge distillation in deep networks](https://arxiv.org/abs/1905.08114)

ICCV_2019 [Data-free learning of student networks](https://arxiv.org/abs/1904.01186)

