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

**Objectives**

1. Improved Personalized Model – for a large majority of the clients
2. Solid Initial Model – some clients have limited or even no data for
3. Fast Convergence – reach a high quality model in small number of training rounds.

### Method

**Definition**: 

For each client $i$, define its local loss function as $L_i(\theta)$

$g_j^i$ be the gradient computed in $j^{th}$  iteration during a local gradient-based optimization process
$$
g FedSGD = \frac{-\beta}{T}\sum_{i=1}^T \frac{\delta L_i(\theta)}{\delta \theta} = \frac{1}{T} \sum_{i=1}^Tg_1^i
$$

$$
\theta_K^i = U_K^i (\theta) = \theta - \beta \sum_{j=1}^K g_j^i  \\
=\theta -\beta \sum_{j=1}^K \frac{\delta L_i(\theta_j)}{\delta \theta}
$$

$$
\frac{\delta U^i_K (\theta)}{\delta \theta} = I - \beta \sum_{j=1}^J \frac{\delta^2 L_i(\theta_j)}{\delta \theta^2}
$$

$$
g MAML = \frac{\delta L_{MAML}}{\delta \theta} = \frac{1}{T} \sum_{i=1}^T 
\frac{\delta L_i(U_K^i(\theta))}{\delta \theta}=\frac{1}{T} \sum_{i=1}^T  
L^{'}_i(U_K^i(\theta))(1-\beta \sum_{j=1}^K \frac{\delta^2 L_I(\theta_j)}{\delta \theta^2})
$$
MAML requires to compute 2nd-order gradients, which can be computationally expensive and creates potentially infeasible memory requirements.


$$
g FedAvg = \frac{1}{T}\sum_{i=1}^T \sum_{j=1}^k g_j^i = \frac{1}{T}\sum_{i=1}^T
g_1^i + \sum_{j=1}^{K-1}\frac{1}{T}\sum_{i=1}^T g^i_{j+1} = gFedSGD + \sum_{j=1}^{K-1}
gFOMAML(j)
$$


1. Run $FedAvg(E)$ with momentum SGD as server optimizer and a relatively larger E. 

2. Switch to $Reptile(K)$ with Adam as server optimizer to fine-tune the initial model. 

3. Conduct personalization with the same client optimizer used during training.

## Think Locally, Act Globally: Federated Learning with Local and Global Representations

### Idea

$(X_m,Y_m)$ represents data on device $m$

$H_m$ are learned local representations via local model $\mathcal{l}_m(\cdot,\theta_m^l):x \rightarrow h$ 

(optional) auxiliary models $a_m(\cdot,\theta_m^a):h \rightarrow z$:

$g(\cdot;\theta^g):h \rightarrow y$ is the global model

AGG is an aggregation function over local updates to the global model.





































