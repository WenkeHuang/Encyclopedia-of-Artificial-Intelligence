###Federated Semi-Supervised Learning With Inter-Client Consistency & Disjoint Learning

#### Problems

Two practical scenarios of Federated Semi-Supervised Learning (FSSL):

- each client learns with only partly labeled data (Labels-at-**Client** scenario) 客户端有数据 用户端同时具备Labeled 和 Unlabeled数据会带来模型对Labeled数据的遗忘
- supervised labels are only available at the **server** 服务器端有数据

![Problem Definition](./img/ICLR_1.PNG)

#### Prior Work

**Federated Learning**.

....

**Semi-Supervised Learning**

The ratio of unlabeled data ($U=\{x_i,y_i\}_{i=1}^U$) is usually much larger than that of the labeled data ($S=\{x_i,y_i\}_{i=1}^S$)  (e.g. 1 : 10). 

Define the $p_{\theta}(y|x)$ be a neural network that is parameterized by weights $\theta$ and predicts softmax outputs $\widehat{y}$ with given input x.

Objective loss function: $\mathcal{l}_{final} \theta)=\mathcal{l}_{s}(\theta)+ \mathcal{l}_{u}(\theta)$.

**Federated Semi-Supervised Learning**

Given a dataset $D=\{x_i,y_i\}_{i=1}^N$ , $D$ is split into a labeleds set $S=\{x_i,y_i\}_{i=1}^S$ and unlabeled data $U=\{x_i,y_i\}_{i=1}^U$.

A global model $G$ and a set of local models $\mathcal{L}$ where

#### Idea

Federated Matching (FedMatch)

- inter-client consistency loss: aims to maximize the agreement between the models trained at different clients 针对不同的用户设计的Loss函数

$$
\frac{1}{H} \sum_{j=1}^H KL [p^{*}_{\theta^{h_j}}(y|u)||p_{\theta^l} (y|u)]
$$

这里$p^{*}_{\theta^{h_j}}(y|u)$代表筛选后的客户端基于模型的相似性，同时星号代表冷冻了这些参数，不更新这些筛选的模型。服务器每次选择并广播$H$个用于帮助的客户端。最终一致约束损失函数：
$$
\Phi(\cdot) = CrossEntropy(\widehat{y},p_{\theta^l}(y|\pi(u)))+\frac{1}{H} \sum_{j=1}^H KL [p^{*}_{\theta^{h_j}}(y|u)||p_{\theta^l}(y|u)]
$$
在这里$\pi(u)$代表随机增强 对于 无标签的数据，而对于对应的生成的标签$\widehat{y}$:
$$
\widehat{y}=Max (\mathbb{I}(p_{\theta^l} (y|u))+\sum_{j=1}^H \mathbb{I}p^{*}_{\theta^{h_j}}(y|u)
$$
$\mathbb{I}$代表生成one-hot的标签，Max($\cdot$) 输出one-hot 标签 中最大参数对应的类结果

- parameter decomposition for disjoint learning: **decomposes** the parameters into one for labeled data and the other for unlabeled data for preservation of reliable knowledge, reduction of communication costs, and disjoint learning 针对平衡对标记数据和非标记数据的记忆能力，把模型参数$\theta$解构为了两部分:$\sigma$ for supervised learning and $\psi$ for unsupervised learning such that $\theta = \sigma + \psi$

有标签数据：
$$
minimize \mathcal{L}_s(\sigma)=\lambda_s CrossEntropy(y,p_{\sigma+\psi^*}(y|x))
$$
无标签数据：
$$
minimize \mathcal{L}_u (\psi) = \lambda _{ICCS}\Phi_{\sigma^*+\psi}(\cdot)+\lambda_{L_2}||\sigma^*-\psi||^2_2+\lambda_{L_1}||\psi||_1
$$

Benefit：

Preservation Reliable Knowledge from Labeled Data

Reduction of Communication Costs

Disjoint Learning

#### 两种场景

**Labels-At-Client Scenario**

“客户端标签”场景假定最终用户会间歇性地注释其本地数据的一小部分（即，占整个数据的5％），而其余数据实例未标记。 这是用户生成的个人数据的常见情况，在这种情况下，最终用户可以轻松地注释数据，但可能没有时间或动力来标记所有数据（例如，为相册或社交网络注释图片中的面孔）。 我们假设客户端对标记和未标记的数据进行训练，而服务器仅聚合来自客户端的更新，然后将聚合的参数重新分发回客户端。

**Labels-At-Server Scenario**

现在，我们描述另一个现实的设置，即服务器标签场景。 此方案假定受监督的标签仅在服务器上可用，而本地客户端使用未标签的数据。

### Personalized Federated Learning With First Order Model Optimization

