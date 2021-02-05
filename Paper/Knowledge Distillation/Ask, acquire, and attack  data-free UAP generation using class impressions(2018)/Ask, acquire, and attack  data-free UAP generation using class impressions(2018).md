# Ask, acquire, and attack: data-free UAP generation using class impressions

## Problems

深度学习模型容易受到输入的具体的噪音（input specific noise），定义为对抗扰动（adversarial perturbations）。

此外，这些对抗性扰动表现出跨模型的普遍性（可转移性）。这意味着相同的对抗扰动样本对不同结构模型都能起作用

存在着input specific noise 叫做通用对抗扰动（Universarial Adversarial Perturbations）-- 当添加时，大多数数据样本可以欺骗目标分类器

两种制造UAPs的方法：

1. data-driven: that require data（需要来自底层数据分布的实际样本，并以高成功率（愚弄）制作UAP）
2. data-free: that do not require data samples（不使用任何数据样本，因此导致较低的成功率）

## Idea

对于data-free的场景，提出了一个方法 使用类印象（class impression）模拟数据样本的效果，以便使用数据驱动的目标来构建UAP。

给定类别和模型下的class impression是属于该类别和模型的样本的泛型表示（在输入空间中）。通过clas impression 用生成模型来制造UAPs。

### Step 1 Ask and Acquire the Class Impressions(生成具有Class Impression的样本)

在我们方法的第一部分中，我们通过简单的优化获得类印象，这些优化可以作为底层数据分布的代表性样本。

学习的参数是训练数据和程序的函数。它们可以被视为模型的记忆，其中训练的本质已经被编码和保存。我们的第一阶段“询问并获取”的目标是挖掘模型的记忆，获取具有代表性的训练数据样本。然后我们只能使用这些代表性的样本来制作uap来欺骗目标分类

we create samples such that the target classifier strongly believes them to be actual samples that belong to categories in the underlying data distri-bution.

换句话说，这些是我们**试图从模型记忆中重建的实际训练数据的印象**。因此我们把它们命名为类印象。产生这些阶级印象的动机是，为了优化愚弄的目标，拥有表现类似于自然数据样本的样本就足够了。

Note that we can create impression ($CI_c$) for any chosen class ($c$) by maximizing the predicted confidence to that class.

输入的是随机的噪声图片基于$U[0,255]$，然后更新图片直到有很高的置信度。

$f$: target classifier ($TC$) under attack, which is a trained model with frozen parameters.

$f_k^i$: $k^{th}$ activation in $i^{th}$ layer of the target classifie

$f^{ps/m}$:output of the pre-softmax layer

$f^{s/m}$:output of the softmax (probability) layer

$v$: additive universal adversarial perturbation (UAP)

$x$: clean input to the target classifier, typically either data sample or class impression

$\xi$: max-norm ($l_1$) constraint on the UAPs, i.e., maximum allowed strength of perturbation that can be added or subtracted at each pixel in the image
$$
CI_c = \mathop{argmax}\limits_{x}f_c^{ps/m}(x)
$$
Typical data augmentations:

1. random rotation in $[-5^\circ,5^\circ]$
2. scaling by a factor randomly selected from $\{0.95, 0.975, 1.0, 1.025\}$,
3. RGB jittering 颜色抖动
4. Random cropping 随机裁剪
5. random uniform noise in $U{[-10,10]}$

### Step2 生成UAP

在获得每个类别的多个类印象之后，我们执行第二部分，即学习生成模型（前馈神经网络）以有效地生成UAP。因此，与现有的解决复杂优化问题以生成UAP的工作不同，我们的方法通过学习的神经网络进行简单的前馈操作。

### Fooling Loss

$G$ takes a random vector $z$ whose components are sampled from a simple distribution (e.g.$ U[−1, 1]$) and transforms it into a UAP via a series of deconvolution layers.

**$G$** : in order to be able to generate the UAPs that can fool the target classifier over the underlying data distribution

clean sample $(x)$

perturbed sample $(x+v)$

**The objective is to make the ‘clean’ and ‘perturbed’ labels different.**

由于softmax的非线性，对其他标签预测的置信度增加，最终导致标签翻转，从而愚弄了目标分类器。
$$
L_f = \log(1-f_C^{s/m}(x+v))
$$

### Diversity loss

Fooling loss 只让$G$ 学会让UAP愚弄目标分类器。但是为了避免学习只能产生一个强UAP的退化G，我们在生成的UAP中引入了多样性。（为了能过针对多个G）

maximizing the pairwise distance between their embeddings $f(x + v_i)$ and $f(x + v_j)$，where $v_i$ and $v_j$ belong to generations within a mini-batch.（也很容易理解，不同的UAP叠加在同一个class impression输出要尽可能不一样）
$$
L_d = \sum_{i.j=1,i \neq j}^K d(f^l(x+v_i),f^l(x+v_j))
$$
K 代表 mini-batch size

d 代表合适的距离度量尺度(Euclidean or cosine distance)

注意，两个嵌入的$f(x+v_i)$和$f(x+v_j)$中两个类印象的$x$是相同的。因此，通过最小化$L_d$将它们分开将使UAPs的$v_i$和$v_j$不同。

Therefore the loss we optimize for training our generative model for crafting UAPs is given by
$$
Loss = L_f +\lambda L_d
$$

## 实验

ILSVRC 数据集 

与现有的数据驱动方法（[13]）类似，每个类使用10个数据样本，我们为每个类提取10个印象，从而生成10000个样本的训练数据

Since our objective is to generate diverse UAPs that can fool effectively, we give equal weight to both the components of the loss, i.e., we keep $\lambda = 1$。

- UAPs and the success rates

与无数据方法FFF相比，由我们的生成网络建模的扰动的成功率更好

提出的方法处理数据缺失的有效性。我们将成功率与数据驱动方法UAP[13]、无数据方法FFF[17]和随机噪声基线进行了比较。

- Comparison with data dependent approaches.

进一步论证了：与最先进的数据驱动方法相比，本文提出的方法构建的扰动的成功率（SR）更高

### Diversity

在损失中包含多样性分量（Ld）的目的是避免学习单个UAP，并且学习能够为给定目标CNN生成多样性UAP集的生成模型。在添加生成的uap之后，我们检查预测标签的分布。这可以揭示是否有一组接收器标签吸引了大多数预测。我们考虑了G学习愚弄VGG-F模型和50000个ILSVRC样本验证集。我们随机选取由G生成的10个UAP，计算出预测标签的平均直方图。对直方图进行排序后，所提出方法的大多数预测标签（95%）分布在1000个目标标签中的212个标签上。而UAP的相同数字是173。**观察到的22.5%的高多样性归因于我们的多样性成分（Ld）。**

## 总结

在本文中，我们提出了一种新颖的方法来**减轻缺乏制作通用对抗扰动（UAP）的数据**。类印象是具有代表性的图像，可以通过从目标模型进行简单的优化轻松获得。通过使用类印象，我们的方法可以**极大地缩小数据驱动方法和无数据方法之间的性能差距，以构建UAP**。成功率更接近于数据驱动的UAP，证明了在制作UAP的过程中班级印象的有效性。查看此观察结果的另一种方式是，有可能以任务特定的方式从模型参数中提取有关训练数据的有用信息。**在本文中，我们提取了类别印象作为代理数据样本，以训练可以为给定目标CNN分类器设计UAP的生成模型。**探索其他应用程序的可行性也是很有趣的。尤其是，我们想调查GAN的现有对抗设置是否可以受益于从鉴别器网络中提取的任何其他信息，并生成更自然的合成数据。我们的方法中提供的生成模型是制作UAP的有效方法。与执行复杂优化的现有方法不同，我们的方法通过简单的前馈操作构造UAP。即使在没有数据的情况下，巨大的成功率，**令人惊讶的跨模型通用性也揭示了当前深度学习模型的严重敏感性**。



