# Federated Model Distillation with Noise-Free Differential Privacy

## Existing Work

**FedMD**

Each model is first trained on the public data to align with public logits, then on its own private data.

**Cronus**

In contrast, Cronus mixes the public dataset (with soft labels) and local private data, then trains local models simultaneously.



Pros: One obvious benefit of sharing logits is the reduced communication costs, without signifi- cantly sacrificing utility. 

Cons: However, both works did not offer theoretical privacy guarantee for sharing model prediction.

## Motivation

Currently, there is no theoretic guarantee that sharing prediction is private and secure. 

Although the privacy concern is mitigated with **random noise perturbation**, it brings a new problem with a substantial **trade-off between privacy budget and model performance**.

## Innovation

- NFDP mechanism: $(\epsilon,\delta)$ - differential privacy of sampling without replacement

$$
(ln \frac{n+1}{n+1-k},\frac{k}{n}) \ -\ differential \ privacy
$$

- NFDP mechanism: $(\epsilon,\delta)$ - differential privacy of sampling with replacement

$$
(kln\frac{n+1}{n},1-(\frac{n-1}{n})^k) \ - \ differential \ privacy
$$



## Contribution

A novel framework called FEDMD-NFDP, which applies the new proposed Noise-Free Differential Privacy (NFDP) mech- anism into a federated model distillation framework.

NFDP can effectively protect the privacy of local data with the least sacrifice of the model utility.

## Baselines

Non-private Federated Model Distillation (FedMD-NP) framework:

各方在其所有本地私有数据上进行训练，协同公共数据的提炼，并使用aggre- gation反馈来更新本地模型，与FedMD相同。需要注意的是，这个框架中没有隐私保障。

Centralized framework: 如果将各方的私密数据集中到一个集中的服务器中，并直接提供给各方。

We use this as the utility upper bound of FedMD-NP and our FEDMD-NFDP.

## Experiments

Each party’s local model is two or three-layer deep neural networks for both MNIST/FEMNIST and CI- FAR10/CIFAR100.

In each communication round, we use a subset of size 5000 that is randomly selected from the entire public dataset. 

## Discussion

**Diversity of public dataset**，公共数据集需要仔细斟酌，甚至需要事先了解各方的私有数据集。未标记的公共数据集的分布可能与各方可用的训练数据的分布相匹配，或者在一定程度上有所不同。当公共数据集与训练数据集的差异越来越大时，差距如何扩大还有待了解，最坏的情况可能是来自不同的领域，没有任何重叠。还有一种可能性是将来自预训练生成器（如GAN）的合成数据作为公共数据，以缓解真实无标签数据集的潜在限制（如获取、存储）。这可能为有效和高效的模型提炼开辟了许多可能性。

**Diversity of local models**

FEDMD-NFDP允许FL中的局部模型不仅在模型结构、大小上有差异，而且在数值精度上也有差异，这对涉及不同硬件配置和计算资源的边缘设备的物联网(Inter- net of Things，IoT)有很大的好处。

**Weighted aggregation**

聚合步骤是基于各方预测的直接平均，即fAggreg对应的是等权重1/N的平均函数，其中N为各方数量。然而，各方对共识的贡献可能不同，特别是在模型和数据异质性的极端情况下。将所有各方分配相同的权重可能会对系统效用产生负面影响。我们指出，有各种聚合算法可以进行更高级的加权平均，以进一步提升效用。这些权重可以用来量化局部模型的贡献，在处理极度不同的模型时发挥重要作用。

**Limitations**

基于NFDP机制的隐私分析，对于有替换或无替换的抽样策略，我们要求每个本地方都有足够大小的数据集。例如，如果每个局部数据只有一个标签，由于私人训练数据的大小限制，我们的机制不能保护任何隐私。

在这种情况下，NFDP可能对三个场景非常有用。
首先，一方需要为他人提供机器学习作为服务（MLaaS），即师生学习框架。虽然这一方包含了大量的私有数据，但NFDP可以帮助它训练一个私有保证模型。第二，一方拥有庞大的数据集，但数据本身缺乏多样性。由于数据多样性的限制，模型仍然不能达到很好的性能。在这种情况下，他们需要与其他人进行通信，以获得更好的模型效用，我们可以在他们的通信过程中使用NFDP来保护他们。最后，一些学习任务只需要一小部分涉及的私有训练数据，比如FedMD。NFDP可以在充分保护隐私的情况下很好地完成这些任务。除了FedMD之外，传统的单次学习和少数短时学习任务也适合使用NFDP进行隐私保护，原因同上。

