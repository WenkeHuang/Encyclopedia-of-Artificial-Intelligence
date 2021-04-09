## Improving Federated Learning Personalization via Model Agnostic Meta Learning

### Problems

FL applications generally face non-i.i.d and unbalanced data available to devices, which makes it challenging to ensure good performance across different devices with a FL-trained global model.

### Contribution

- The popular FL algorithm, Federated Averaging, can be interpreted as a meta learning algorithm. FedAvg可以被解释成元学习算法

- Careful fine-tuning can yield a global model with higher accuracy, which is at the same time easier to personalize. However, solely optimizing for the global model accuracy yields a weaker personalization result.  仔细的微调可以生成具有更高准确度的全局模型，同时更易于个性化。 但是，仅针对全局模型精度进行优化会产生较弱的个性化结果

- A model trained using a standard datacenter optimization method is much harder to personalize, compared to one trained using Federated Averaging, supporting the first claim. 与使用联邦平均法训练的模型相比，使用标准数据中心优化方法训练的模型更难个性化

### Existing Methods

现有的FL个性化工作直接采用融合的初始模型，并通过梯度下降进行个性化评估

### Idea

We refer to a trained global model as the initial model, and the locally adapted model as the personalized model.

