# Multi-Center Federated Learning

## Motivation

现有的联合学习方法通常采用单个全局模型来通过汇总其梯度来捕获所有用户的共享知识，而不管其数据分布之间的差异如何。

但是，由于用户行为的多样性，将用户的梯度分配给不同的全局模型（即中心）可以更好地捕获用户之间数据分布的异质性。

## Methodology

### Multi-center Aggregation

原始联合学习使用一种中央模型将聚合结果存储在服务器中，这称为单中心聚合。 用户的行为数据通常是多种多样的，这意味着最佳的本地模型将有所不同

In our proposed method, all local models will be grouped to $K$ clusters, which is denoted as $C_1$, · · · , $C_K$. Eachc luster $C_k$ consists of a set of local model parameters $\{W_j\}^{m_k}_{j=1}$, and corresponding
center model $\widetilde{W}(k)$.

### Objective Function

在联盟学习的一般设置中，学习目标是最大程度地减少设备上所有监督学习任务的总损失

模型集合聚合机制是一种随机梯度下降（SGD）程序，可调整中央模型的参数以接近局部模型的参数。 但是，SGD训练过程基于以下假设：所有分布式数据集均从一个IID来源中提取，而对非IID数据的训练是联合学习最吸引人的特征。

To tackle the non-IID challenge in federated learning, we propose 

1. **distance-based federated loss** – a new objective function using a distance between parameters from the global and local models.

$$
\mathcal{L}=\frac{1}{m}\sum_{i=1}^m Dist(W_i,\widetilde{W})
$$

$$
Dist(W_i,\widetilde{W}) \triangleq = ||W_i-\widetilde{W}||^2
$$

2. **multi-center federated loss** – the total distance-based loss to aggregate local models to multiple centers.

$$
\mathcal{L} = \frac{1}{m}\sum_{k=1}^K\sum_{i=1}^mr_i^{(k)}Dist(W_i,\widetilde{W}^{(k)})
$$

### Optimization Method

1.  E-step – updating cluster assignment $r^{(k)}_i$ with fixed $W_i$

Firstly, for the **E-Step**, we calculate the distance between the cluster center and nodes – each node is a model’s parameters $W_i$, then update the cluster assignment $r^{(k)}_i$ by
$$
r_i^{k}= \begin{cases}
      1, if k= argmin_{j}Dist(W_i,\widetilde{W}^{(j)})\\
      0, otherwise
  \end{cases}
$$

2. M-step – updating cluster centers $\widetilde{W}^{(k)}$

$$
\widetilde{W}^{(k)}=\frac{1}{\sum_{i=1}^mr_i^{(k)}}\sum_{i=1}^m r_i^{(k)}W_i
$$

3. updating local models by providing new initialization $\widetilde{W}^{(k)}$

The global model’s parameters $\widetilde{W}^{(k)}$ are sent to each device in cluster k to update its local model, and then we can finetune the local model’s parameters $W_i$ using a supervised learning algorithm on its own private training data.