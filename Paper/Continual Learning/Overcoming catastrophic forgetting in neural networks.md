# Overcoming catastrophic forgetting in neural networks

## Problem

实现通用人工智能需要模型能够学习和记忆许多不同的任务。这在现实环境中尤其困难：任务序列可能没有明确的标记，任务可能无法预测地切换，任何单个任务可能不会长时间重复。因此，至关重要的是，模型必须表现出**Continual Learning**持续学习的能力：即在不忘记如何执行以前训练过的任务的情况下学习连续任务的能力。

Continual learning poses particular challenges for artificial neural networks due to the tendency for knowledge of previously learnt task(s) (e.g. task A) to be abruptly lost as information relevant to the current task (e.g. task B) is incorporated. This phenomenon, termed ***catastrophic forgetting***.

## Related Work

- Current approaches have typically ensured that data from all tasks are simultaneously available during training.

如果任务是按顺序呈现的，那么多任务学习只能在数据被一个情景记忆系统记录并在训练过程中回放到网络上时才能使用。这种方法对于学习大量任务是不切实际的，因为在我们的设置中，它要求存储和回放的内存量与任务数量成比例。

- humans and other animals appear to be able to learn in a continual fashion

与人工神经网络形成鲜明对比的是，人类和其他动物似乎能够持续学习。最近的证据表明，哺乳动物的大脑可以通过保护大脑皮层回路中先前获得的知识来避免灾难性遗忘。当老鼠掌握了一项新技能时，一部分兴奋性突触得到加强；这表现为神经元单个树突棘的体积增加。关键的是，尽管随后学习了其他任务，这些增大的树突棘仍然存在，这说明了几个月后的绩效保持。当这些刺被选择性地“擦除”时，相应的技能就会被遗忘。这提供了因果证据，**支持保护这些强化突触的神经机制对保持任务绩效至关重要**。总之，这些实验结果以及神经生物学模型表明哺乳动物新皮质的持续学习依赖于任务特异性突触巩固的过程，因此，关于如何执行先前获得的任务的知识被持久地编码在一定比例的突触中，这些突触的可塑性变差，因此在长时间内是稳定的。

## Elastic weight consolidation

**Synaptic consolidation** enables continual learning by reducing the plasticity of synapses that are **vital** to previously learned tasks

### Assumption

许多模型参数的不同设置可以带来相同的表现能力，这和EWC是相关的，over-parameterization让学习任务B的时候可以让模型依旧接近任务A的参数设置。因而EWC可以保护在任务A上的表现能力通过约束参数在一个固定的区间。

![EWC](./img/EWC.png)

灰色和黄色分别是对应旧任务和新任务的error surface。这里为了示意，在最终收敛参数周围，使用了二次近似，在平面上，即为一个椭圆。如果我们在Finetune的时候不加入任何约束，那么最终收敛的参数自然会到新任务loss最低的地方，即蓝色箭头所指方向。但是显然，这样的结果对于旧任务是一个很差的参数。如果我们能够充分使用局部error surface的信息，使参数收敛到上图中两个椭圆重叠的部分，那必然是一个最好的结果。这也就是[4]这个工作想要达到的目的。具体做法上，作者使用了Fisher Information Matrix来近似Hessian Matrix，并为了效率考虑，只选取了对角线元素做近似

*This constraint is implemented as a quadratic penalty, and can therefore be imagined as a spring anchoring the parameters to the previous solution, hence the name elastic. Importantly, the stiffness of this spring should not be the same for all parameters; rather, it should be greater for those parameters that matter most to the performance during task A*

### Idea

In order to justify this choice of constraint and to define which weights are most important for a task, it is useful to consider neural network training from a probabilistic perspective. **From this point of view, optimizing the parameters is tantamount to finding their most probable values given some data D.**

**Bayes's Rule**
$$
p(\theta|D)=\frac{p(D|\theta) \cdot p(\theta)}{p(D)}
$$

$$
log \ p(\theta|D)= log \ p(D|\theta) + log \ p(\theta)-log \ p(D)
$$


Hence
$$
log \ p(\theta|D)= log \ (p(D_B|\theta) + log \ p(\theta|D_A)-log \ p(D_B)
$$
Note that the left hand side is still describing the posterior probability of the parameters given the entire dataset, while the right hand side only depends on the loss function for task B $log \ p(D_B|\theta)$

$p(\theta|A)$ : This posterior probability must contain information about **which parameters were important to task A and is therefore the key to implementing EWC.**
$$
\mathcal{L}(\theta) = \mathcal{L}_B(\theta)+\sum_i \frac{\lambda}{2}F_i(\theta_i-\theta^*_{A,i})^2 
$$
$\mathcal{L}_B(\theta)$ is the loss for task B only, $\lambda$ sets how important the old task is compared to the new one and $i$ labels each parameter



以往解决深度神经网络连续学习问题的方法都依赖于对网络超参数的仔细选择，以及其他标准的正则化方法，以减轻灾难性遗忘。然而，在这项任务中，他们只在最多两个随机排列上取得了合理的结果。使用类似的交叉验证超参数搜索，我们将传统的dropout正则化与EWC进行了比较。我们发现，随机梯度下降与辍学正则化单独是有限的，它不能扩展到更多的任务。相反，EWC允许大量的任务按顺序学习，错误率只有适度的增长。

## Discussion

我们提出了一种新的算法，弹性权值合并，解决了神经网络的连续学习问题。EWC允许在新的学习过程中保护先前任务的知识，从而避免对旧能力的灾难性遗忘。它通过选择性地降低重量的可塑性来实现，因此与突触巩固的神经生物学模型相似。我们将EWC作为一个软的二次约束来实现，即每个权重都被拉回到原来的值，其大小与其对先前学习任务的重要性成比例。在任务共享结构的程度上，使用EWC训练的网络重用网络的共享组件。我们进一步证明，EWC可以有效地与深度神经网络相结合，以支持具有挑战性的强化学习场景中的持续学习，例如Atari 2600游戏。









=
