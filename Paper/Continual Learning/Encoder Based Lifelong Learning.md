# Overcoming catastrophic forgetting in neural networks

## Problem

The main challenge that vision systems face in this context is catastrophic forgetting: as they tend to adapt to the most recently seen task, they lose performance on the tasks that were learned previously.

## Target

In this work we aim at preserving the knowledge of the previous tasks and pos- sibly benefiting from this knowledge while learning a new task, without storing data from previous tasks.

## Related Work

- Learning without forgetting (LwF)

LWF proposes to preserve the previous performance through the knowledge distillation loss.

☑️ This method reduces the forgetting, especially when the datasets come from related manifolds.

❌ LwF suffers from **a build up of errors** in a sequential scenario where the data comes from the same environment.

❌ LwF performance drops when the model is exposed to a sequence of tasks drawn from differ- ent distributions.

- iCaRL: Incremental Clas- sifier and Representation Learning

iCaRL proposes to store a selection of the previous tasks data to overcome this issue – **something we try to avoid.**

- EWC

☑️ The use of the Fisher matrix prevents the weights that are important for the first task to change much

❌ First, the method keeps the weights in a neighborhood of one possible mini- mizer of the empirical risk of the first task. However, there could be another solution that can give a better compromise between the two tasks.

❌ Second, it needs to store a large number of parameters that grows with the total number of weights and the number of tasks.

*For these reasons, rather than constraining the weights, we **choose to constrain the resulting features**, enforcing that those that are important for the previous tasks do not change much. By constraining only a sub-manifold of the features, we allow the weights to adjust so as to optimize the features for the new task, while preserving those that ensure a good performance on the previous tasks.*

## Idea

In practice, the empirical risk is minimized:
$$
R_N = \frac{1}{N}\sum_{i=1}^N(\mathcal{l}(T_T \circ T \circ F(X_i^{(\mathcal{T})}),Y_i^{(\mathcal{T})})+\sum_{t=1}^{\mathcal{T}-1}\mathcal{l}_{dist}(T_t \circ T \circ F(X_i^{(\mathcal{T})}),T_t^* \circ T^* \circ F^*  \\ (X_I^{(\mathcal{T})})),+\sum_{t=1}^{\mathcal{T}-1}\frac{\alpha_t}{2}||\sigma(W_{enc,t}F(X_i^{(\mathcal{T})})-\sigma W_{enc,t}F^*(X_i^{(\mathcal{t})}))||_2^2)
$$
Autoencoder:
$$
arg \min_{\gamma} \mathbb{E}_{x^{(1)},y^{(1)}}[\lambda||r(F^*(x^{(1)}))-F^*(x^{(1)})||_2+\mathcal{l}(T_1^* \circ T^*(r(F^*(x^{(1)}))),y^{1})]
$$


## Discussion

这里给出的解决方案减少了对早期记忆的遗忘，通过控制不同任务的表示之间的距离。我们没有保留以前任务的最佳权重，而是提出了一种在相应环境中保留对性能至关重要的特征的替代方法。欠完备自动编码器用于学习表示这些重要特征的子流形。该方法以两个、三个或五个任务为序列，从一个小的或一个大的数据集开始，对图像分类问题进行了测试。在所有测试场景中，性能都比最新技术有所提高。特别是，我们展示了更好的保存旧任务。尽管有明显的改进，但这项工作确定可能的进一步发展。一个值得探索的方向是使用自动编码器作为数据发生器，而不是依赖新的数据。在新数据不能很好地表示以前的分布的情况下，这将提供一个更强有力的解决方案。







=
