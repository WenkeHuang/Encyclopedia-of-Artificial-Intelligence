## Group Knowledge Transfer: Federated Learning of Large CNNs at the Edge

### Abstract

例如，尽管由于FL的机密性和机密性，对FL的实际需求很大，但是联合学习（FL）可能会给边缘节点的计算能力带来负担。FedGKT设计了另一种最小化方法的变体 ，在边缘前端上训练小型CNN，并通过知识注入将其知识转移到大型服务器端CNN。FedGKT将多个优势整合到一个框架中：减少了对边缘计算的需求，减少了大型CNN的通信距离， 数据集（CIFAR-10，CIFAR-100和CINIC-10）及其非IID变体使用ResNet-56和ResNet-110训练CNN进行并发同步训练，同时保持与FedAvg相当的模型精度。 我们的结果表明，FedGKT和FedAvg具有可比甚至更高的精度。 

### Problems

为了解决边缘节点的计算限制，基于模型并行度的拆分学习（SL）对大型模型进行分区，并将神经体系结构的某些部分卸载到云中，但是SL存在严重的散乱问题，因为单个小批处理迭代需要多个 服务器与边缘之间的轮次通信

### Idea

a small feature extractor model $W_e$ 

a large-scale server-side model $W_s$

put them on the edge and the server, respectively. We also add a classifier $W_c$ for $W_e$ to create a small but fully trainable model on the edge.

Consequently, we reformulate a single global model optimization into an non-convex optimization problem that requires us to solve the server model $F_s$ and the edge model $F_c$ simultaneously. Our reformulation is as follows:
$$
\mathop{argmin}_{W_s}F_s(W_s,W_e^*) = \mathop{argmin}_{W_s} \sum_{k=1}^K \sum_{i=1}^{N^{(k)}} \mathcal{l}_s(f_s(W_S;H_i^{(k)}),y_i^{(k)})
$$

$$
subject \ to: H_i^{(k)} = f_e^{(k)} (W_e^{(k)};X_i^{(k)})
$$

$$
\mathop{argmin}_{(W_e^{(k)},W_c^{(k)}}F_c(W_e^{(k)},W_c^{(k)}) = \mathop{argmin}_{W_e^{(k)},W_c^{(k)}} \sum_{i=1}^{N^{(k)}} \mathcal{l}_c(f^{(k)}((W_e^{(k)},W_c^{(k)});X_i^{(k)}),y_i^{(k)})
$$

## Federated Principal Component Analysis

### Idea

In this work we pursue a combined federated learning and differential privacy framework to compute PCA in a decentralised way and provide precise guarantees on the privacy budget.

### Contribution

在这项工作中，我们介绍了一种用于计算PCA的联邦流和差分私有算法。 我们的算法从多个方面改进了最新技术：它是与时间无关的，异步的和差分私有的。 通过扩展到流式传输和非对称设置，可以保证DP。 我们在这样做的同时保留了MOD-SuLQ提供的几乎最佳的渐近保证。 我们的算法得到了一些理论结果的补充，这些理论结果保证了有限的估计误差和对数据排列的鲁棒性。 我们通过大量的数值实验对我们的工作进行了补充，这些实验表明，在收敛性，有界估计误差和低内存需求方面，Federated-PCA与其他方法相比具有优势。 **未来工作的一个有趣途径是研究联合PCA在设置缺失值的同时保留差异性隐私。**

## Personalized Federated Learning with Moreau Envelopes

### Problems

One challenge associated with FL is **statistical diversity** among clients, which restricts the global model from delivering good performance on each client’s task.

> How can we leverage the global model in FL to find a “personalized model” that is stylized for each client’s data?

### Idea

we propose an algorithm for personalized FL (pFedMe) using Moreau envelopes as clients’ regularized loss functions, which help decouple personalized model optimization from the global model learning in a bi-level problem stylized for personalized FL.
$$
f_i(\theta_i) +\frac{\lambda}{2}||\theta_i-w||^2
$$
$\theta_i$ denotes the personalized model of client $i$ and $\lambda$ is a regularization parameter that controls the strength of $w$ to the personalized model.

Large $\lambda$ can benefit clients with unreliable data from the abundant data aggregation.

Small $\lambda$ helps clients with sufficient useful data prioritize personalization.


$$
\mathop{min}_{w\in R^d}\{F(w):=\frac{1}{N}\sum_{i=1}^NF_i(w)\} \\
where \ F_i(w) = \mathop{min}_{\theta_i \in R^d} \{f_i(\theta_i)+\frac{\lambda}{2}||\theta_i-w||^2\}
$$
Overall, the idea is allowing clients to pursue their own models with different directions, but not to stay far away from the “reference point” $w$, to which every client contributes.



In pFedMe, while $w$ is found by exploiting the data aggregation from multiple clients at the outer level, $θ_i$ is optimized with respect to (w.r.t) client i’s data distribution and is maintained a bounded distance from w at the inner level.

The optimal personalized model, which is the unique solution to the inner problem of pFedMe and also known as the proximal operator in the literature, is defined as follows:
$$
\widehat{\theta}_i:=prox_{f_i/\lambda}(w)=\mathop{argmin}_{\theta_i \in R^d}\{f_i(\theta_i)+\frac{\lambda}{2}||\theta_i-w||^2\}
$$
Compared to Per-FedAvg, our problem has a similar meaning of w as a “meta-model”, but instead of using $w$ as the initialization, **we parallelly pursue both the personalized and global models by solving a bi-level problem**, which has several benefits.

First, while Per-FedAvg is optimized for **one-step gradient update** for its personalized model, pFedMe is **agnostic** to the inner optimizer, which means it can be solved using any iterative approach with multi-step updates. Second, by re-writing the personalized model update of Per-FedAvg as:
$$
\theta_i(w) = w -\alpha \nabla f_i(w) = \mathop{argmin}_{\theta_i \in R^d}\{<\nabla f_i(w),\theta_i-w>+\frac{1}{2\alpha}||\theta_i-w||^2\}
$$

## Lower Bounds and Optimal Algorithms for Personalized Federated Learning

### Contribution

Our first contribution is establishing the first lower bounds for this formulation, for both the communication complexity and the local oracle complexity.

Our second contribution is the design of several optimal methods matching these lower bounds in almost all regimes.

## Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach

我们考虑了异构情况下的联合学习（FL）问题，并研究了经典FL公式的个性化变体，其中我们的目标是为用户找到合适的初始化模型，以快速适应每个用户的本地数据 在训练阶段之后。 

我们强调了此公式与不可知模型元学习（MAML）的联系，并展示了如何将MAML的分散实现（我们称为Per-FedAvg）用于解决所提出的个性化FL问题。 

我们还描述了在非凸设置中实现一阶最优的Per-FedAvg的整体复杂性。 

最后，我们提供了一组数值实验来说明Per-FedAvg的两种不同的一阶逼近的性能以及它们与FedAvg方法的比较，并表明与 FedAvg的解决方案。

## Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization

本文提出了一个框架来分析现有的FL优化算法（例如 FedAvg和FedProx）在data heterogeneous情况下的收敛性，并且提出了FedNova算法 - a normalized averaging algorithm that eliminates objective inconsistency while preserving fast error convergence.











































