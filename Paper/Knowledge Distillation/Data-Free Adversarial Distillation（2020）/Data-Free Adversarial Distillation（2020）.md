##Data-Free Adversarial Distillation

### Problems

- the training data of released pretrained models are often unavailable due to privacy, transmission, or legal issues（老师数据隐私不给看）
-  One strategy to deal with this problem is to use some alternative data, but it leads to a new problem where users are utterly ignorant of the data domain, making it almost impossible to collect similar data（相似数据存在domain gap）
- 同时，即使域信息是已知的，收集大量数据仍然是繁重且昂贵的。 在这种情况下，另一个折衷的策略是使用一些无关的数据进行训练。 但是，由于产生的数据偏差，它极大地降低了学生的表现。 （无关数据会降低模型表现）
- 已有的工作：Zero-shot knowledge distillation in deep networks和Data-free learning of student networks 存在问题：For example, their generation constraints are empirically designed based on assumption that an appropriate sample usually has a high degree of confidence in the teacher model. Actually, the model maps the samples from the data space to a very small output space, and a large amount of information is lost. It is difficult to construct samples with a fixed criterion on such a limited space.（例如，根据以下假设凭经验设计它们的生成约束：适当的样本通常在教师模型处有很高的置信度。 实际上，该模型将样本从数据空间映射到很小的输出空间，并且会丢失大量信息。 在如此有限的空间上，很难以固定的标准构造样本。）

### Idea

这篇文章中的算法更接近传统的GAN，并且声称在图像分割任务中也得到了与data-driven方法接近的效果。

训练过程也基本与传统的GAN一致，因为这个模型中的判别器由Teacher和Student共同组成，即判别器的参数不是固定的，下面简述一下训练过程：

#### 模拟阶段

固定Generator， 只更新Discriminator（其实只是更新Student部分）

使用Teacher和Student之间的MAE误差作为损失函数。

这一阶段与常规的知识蒸馏方法类似，作者认为这一步的意义除了向Teacher学习之外，还有一个意义就是构建更好的搜索空间，迫使生成器去寻找新的困难样本。

#### 生成阶段

固定Discriminator， 只更新Generator

**这一步需要让Generator去学习如何生成更难分辨的样本，也就是需要提高Teacher和Student的输出结果的差异性**，损失函数仍然选择了两个模型之间的MAE，只是加了个负号。

后来作者发现使用这个loss函数训练得到的生成样本都很奇怪，对这个MAE损失做了负对数处理，得到了比较好的效果。

### Related Work

除了利用GAN做无数据蒸馏的方法外，还有一些利用反向传播更新输入数据的方式来获取与训练数据相似的样本的算法（方法类似于图片风格化算法），这方面的算法也有两篇比较典型的论文，分别是利用训练数据产生的激活值以及Batchnorm参数来进行反向传播更新输入噪声：

Data-Free knowledge distillation for Deep Neural Networks

Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion