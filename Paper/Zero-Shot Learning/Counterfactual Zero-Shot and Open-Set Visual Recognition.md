**Author**:  Zhongqi Yue, Tan Wang, Hanwang Zhan, Qianru Sun, Xian-Sheng Hua

**Conference**: CVPR 2021

## Idea

现有的零次学习和开集识别中，见过和未见过类别间识别率的严重失衡。这种失衡是因为对未见过类别样本失真的想象。提出了一种反事实框架，通过基于样本特征的反事实生成保真

**基于因果的生成模型Generative Causal Model (GCM)**

对于一个图片 ![[公式]](https://www.zhihu.com/equation?tex=x) ，我们通过encoder ![[公式]](https://www.zhihu.com/equation?tex=z%28%5Ccdot%29) 拿到这个图片的样本特征 ![[公式]](https://www.zhihu.com/equation?tex=Z%3Dz%28x%29) （比如front-view，walking等），基于这个样本特征 ![[公式]](https://www.zhihu.com/equation?tex=Z)（fact）和不同的类别特征 ![[公式]](https://www.zhihu.com/equation?tex=Y) （counterfact），我们可以生成不同类别的反事实图片 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7Bx%7D) （front-view,walking的猫，羊和鸡等等）。直觉上我们知道，因为反事实生成的猫、羊和鸡的图片和 ![[公式]](https://www.zhihu.com/equation?tex=x) 不像， 那么 ![[公式]](https://www.zhihu.com/equation?tex=x) 肯定不属于这三个类别。这种直觉其实是有理论支持的---叫做**反事实一致性（Counterfactual Consistency Rule)**，通俗的解释就是counterfact和fact重合时，得到的结果就是factual的结果，比如fact是昨天吃冰淇凌拉肚子，那么反事实问题“如果我昨天吃冰淇凌会怎么样呢？”的答案就是拉肚子。

（**反事实生成和现有生成模型的最大区别就是基于了特定的样本特征** ![[公式]](https://www.zhihu.com/equation?tex=Z%3Dz) **（fact）来进行生成，而非高斯噪声**。)

那么我们就能够用样本空间当中的量度去比较 ![[公式]](https://www.zhihu.com/equation?tex=x) 和生成的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7Bx%7D) ，从而用consistency rule判断 ![[公式]](https://www.zhihu.com/equation?tex=x) 是属于见过的还是没见过的类。



对于Zero Shot

1. 用未见过类别的attribute 生成反事实样本，然后用训练集的样本（见过的类）和生成的样本（未见过的类）训练一个线性分类器，
2. 对输入样本 ![[公式]](https://www.zhihu.com/equation?tex=X%3Dx) 进行分类后，取见过类和未见过类概率的top-K的平均值。
3. 如果未见过类上的平均值较小，我们就认为样本 ![[公式]](https://www.zhihu.com/equation?tex=X%3Dx) 不像未见过的类（not consistent)，把这个样本标注成属于见过的类，并使用在见过类的样本上面监督学习的分类器来分类

对于开集任务：

1. 在OSR里面，因为没有未见类别的信息，我们用见过类的one-hot label，作为 ![[公式]](https://www.zhihu.com/equation?tex=Y) 生成反事实样本
2. 如果 ![[公式]](https://www.zhihu.com/equation?tex=x) 和生成的样本在欧式距离下都很远（not consistent），就认为 ![[公式]](https://www.zhihu.com/equation?tex=x) 属于未见过的类，并标为“未知”，反之则用监督学习的分类器即可。



**核心要求是生成保真的样本** -- **保真生成的充要条件是样本特征和类别特征之间解耦（disentangle）**

1. ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta) -VAE loss：这个loss要求encode得到的 ![[公式]](https://www.zhihu.com/equation?tex=Z%3Dz%28x%29) ，和样本自己的 ![[公式]](https://www.zhihu.com/equation?tex=Y%3Dy) ，可以重构样本 ![[公式]](https://www.zhihu.com/equation?tex=X%3Dx) ，并且encode出来的 ![[公式]](https://www.zhihu.com/equation?tex=Z) 要非常符合isotropic Gaussian分布。这样通过使 ![[公式]](https://www.zhihu.com/equation?tex=Z) 的分布和 ![[公式]](https://www.zhihu.com/equation?tex=Y) 无关实现解耦；
2. Contrastive loss：反事实生成的样本中， ![[公式]](https://www.zhihu.com/equation?tex=x) 只和自己类别特征生成的样本像，和其他类别特征生成的样本都远。这个避免了生成模型只用 ![[公式]](https://www.zhihu.com/equation?tex=Z) 里面的信息进行生成而忽略了 ![[公式]](https://www.zhihu.com/equation?tex=Y) ，从而进一步的把 ![[公式]](https://www.zhihu.com/equation?tex=Y) 的信息从 ![[公式]](https://www.zhihu.com/equation?tex=Z) 里解耦；
3. GAN loss：这个loss直接要求反事实生成的样本被discriminator认为是真实的，通过充要条件，用保真来进一步解耦。

## Drawback & Tips

对样本解耦表示，之前有碰到过类似的，解耦前景content和背景background 用于Domain Gap的处理，

解耦的存在 能够更好地进行因果推理

或许现在的工作我们做的KL 分布，其实参与者是不懂这个东西的，就类似于在错误的上面进行学习，emmmmm 想不明白

## Link

[[CVPR 2021] 让机器想象未见的世界：反事实的零次和开集识别 - 斑头雁的文章 - 知乎 ](https://zhuanlan.zhihu.com/p/365089242)

