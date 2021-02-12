# Born-Again Neural Networks

## Problems

知识蒸馏将知识从一个复杂的机器学习模型迁移到另一个紧凑的机器学习模型，而一般紧凑的模型在性能上会有一些降低。本文探讨了同等复杂度模型之间的知识迁移，并发现知识蒸馏中的学生模型在性能上要比教师模型更强大。

**训练一个和teacher参数一样多的student网络，并且准确率超过了teacher网络**，提出了两种蒸馏方法：（1）通过teacher max加权置信度（Confidence-Weighted by Teacher Max (CWTM)）（2）打乱非预测类别的概率分布（Dark Knowledge with Permuted Predic-tions (DKPP).）。这两种方法都用到了KD的组成成分，描述了teacher网络的输出在预测类别和非预测类别上的影响。

## Existing Work

知识蒸馏带来的梯度分为两个部分：

- a dark knowledge term, containing the information on the wrong outputs,
- a ground-truth component which corresponds to a simple rescaling of the original gradient that would be obtained using the real labels.（我们将第二个术语解释为基于教师模型对重要样本的最大置信度，使用每个样本的重要性权重和对应的真实标签进行训练。这说明了 KD 如何在没有暗知识的情况下改进学生模型。）

$$
\theta_1^* = \mathop{argmin}_{\theta_1} \mathcal{L}(y,f(x,\theta_1))
$$



## Idea

![BAN_structure](https://github.com/WenkeHuang/Encyclopedia-of-Artificial-Intelligence/blob/master/Paper/Knowledge%20Distillation/Born-Again%20Neural%20Netowrk%20(2018)/BAN_structure.png)

图 ：BAN 训练过程的图形表示：第一步，从标签 Y 训练教师模型 T。然后，在每个连续的步骤中，**从不同的随机种子初始化有相同架构的新模型**，并且在**前一学生模型**的监督下训练这些模型。在该过程结束时，通过多代学生模型的集成可获得额外的性能提升。

Born-Again Networks（BAN）基于知识蒸馏/模型压缩论文中证明的经验发现，可以通过修改损失函数来减少泛化误差。 
$$
\mathcal{L}(f(x,\mathop{aig \ min}_{\theta_1}\mathcal{L}(y,f(x,\theta_1))),f(x,\theta_2)
$$
Additionally, we present experiments addressing the case when the teacher and student networks have similar ca- pacity but different architectures.

### Sequence of Teaching Selves Born-Again Networks Ensemble

我们通过多代知识转移顺序应用BAN。 在每种情况下，都要训练第k个模型，并从第k-1个学生那里转移知识：
$$
\mathcal{L}(f(x,\mathop{aig \ min}_{\theta_{k-1}}\mathcal{L}(y,f(x,\theta_{k-1}))),f(x,\theta_k)
$$
最后，类似于重新启动后合并SGD的多个snapshots，我们通过平均多代BAN的预测来生成Born-Again Network Ensembles（BANE）。
$$
\widehat{f}^k(x)=\sum_{i=1}^kf(x,\theta_i)/k
$$

### Dark Knowledge Under the Light

- 软标签的熵比硬标签的熵大，包含了更丰富的信息概率输入包含了输入与这些类别的相似性信息这些信息有正则化效果

$$
\frac{\partial \mathcal{L}_i}{\partial \mathcal{z}_i}=q_i-p_i=\frac{e^{z_i}}{\sum_{j=1}^ne^{z_j}}-\frac{e^{t_i}}{\sum_{t=1}^ne^{t_j}}
$$

- 从硬标签和软标签出发，比较流入目标类别的神经元的梯度，可以看出知识蒸馏可能对样本加入了重要性权重，这里的重要性来自teacher网络的目标类别输出的概率。

$$
\frac{\partial \mathcal{L}_*}{\partial \mathcal{z}_*}=q_*-y_*=\frac{e^{z_*}}{\sum_{t=1}^ne^{z_j}}-1
$$

When the loss is computed with respect to the complete teacher output, the student back-propagates the mean of the gradients with respect to correct and incorrect outputs across a**ll the b samples s of the mini-batch** (assuming without loss of generality the $n$th label is the ground truth label $∗$):
$$
\sum_{s=1}^b \sum_{i=1}^b \frac{\partial \mathcal{L}_{i,s}}{\partial \mathcal{z}_{i,s}} = \sum_{s=1}^b (q_{*,s}-p_{*,s})+\sum_{s=1}^b \sum_{i=1}^{n-1}(q_{i,s}-p_{i,s})
$$
up to a rescaling factor $1/b$.

在一个大小为b的batch中，对所有输出神经元（softmax之前）的梯度和可以分解为两部分：左侧为目标类别的梯度，右侧为通过dark knowledge得到的非目标类别的梯度

The **second** term corresponds to the information incoming from all the wrong outputs, via dark knowledge.

The **first** term corresponds to the gradient from the correct choice and can be rewritten as:
$$
\frac{1}{b}\sum_{s=1}^b(q_{*,s}-p_{*,s}y_{*,s})
$$
当老师模型正确且对自己的输出充满信心时，即$p_{∗,s} \approx 1$，减小真实标签梯度。 置信度较低的样本的梯度被重新缩放了一个因子$p_{*，s}$，但对总体训练信号的贡献却降低了。
$$
\sum_{s=1}^b \frac{w_s}{\sum_{u=1}^b w_u}(q_{*,s}-y_{*,s}) =\sum_{s=1}^b \frac{p_{*,s}}{\sum_{u=1}^b p_{*,u}}(q_{*,s}-y_{*,s}) 
$$

>  所以会不会KD的提升不在于或者不仅仅在于非目标类别的输出所包含的信息，KD的提升仅仅就是一个样本重要性的体现？

### 两种对比idea

**Confidence Weighted by Teacher Max(CWTM)**
$$
\sum_{s=1}^b \frac{max \ p_{.,s}}{\sum_{u=1}^b max \ p_{.,u}}(q_{*,s}-y_{*,s})
$$
将式中的权重系数用teacher网络输出的最大值代替，这样就可以用普通的交叉熵加权重来训练

**Dark Knowledge with Permuted Predictions(DKPP)**

将teacher网络中**不是最大值的输出打乱**，并改为teacher网络的最大值，得到的梯度，$\phi(p_{j,s})$表示teacher网络非最大值打乱后的结果
$$
\sum_{s=1}^b \sum_{i=1}^n = \sum_{s=1}^b(q_{*,s}-max\ p_{.,s})+\sum_{s=1}^b\sum_{i=1}^{n-1} q_{i,s}-\phi(p_{j,s})
$$

## 实验

探索BAN在网络的深度和宽度改变时的稳定性

探索不同类别的网络间的BAN（ResNets teacher，DenseNets student；DenseNets student，ResNets teacher）

![BAN_Figure](https://github.com/WenkeHuang/Encyclopedia-of-Artificial-Intelligence/blob/master/Paper/Knowledge%20Distillation/Born-Again%20Neural%20Netowrk%20(2018)/BAN_Figure.png)

从Table 2左侧表的右边两行可以看出，尽管CWTM,DKPP没有 dark knowledge 的参与，但是这两种模型仍然稳定的提升了模型的泛化能力,这说明KD的提升不仅仅来自于非目标类别的信息.DKPP的结果说明，尽管打乱了所有非最大值的排序（丢失了非目标类别的信息），但只靠最大类别的概率仍然可以稳定地提升模型的泛化能力CWTM完全去掉了非目标类别的信息，但仍然可以提升模型的效果，表明KD有效的原因还可以部分归咎于他对目标类别的损失有重要性加权的意思（teacher集中于max则权重大，否则权重小.

## Conclusion

有人对人类发展做出分析，得到了一种序列化自我学习机制的思想,Minsky认为人类童年时代智力突然的提升应该归因于长期隐藏的以前的自己对现在学习过程的教导,Minsky认为我们的长期自我感知是通过集成的多代的内部学习过程建造的.本文的实验结果表明，这样的知识迁移在人工神经网络中也是成功的.