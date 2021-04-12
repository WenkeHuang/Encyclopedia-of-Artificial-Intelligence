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































































