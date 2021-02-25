###FEDERATED SEMI-SUPERVISED LEARNING WITH INTER-CLIENT CONSISTENCY & DISJOINT LEARNING

#### Problems

Two practical scenarios of Federated Semi-Supervised Learning (FSSL):

- each client learns with only partly labeled data (Labels-at-**Client** scenario) 客户端有数据
- supervised labels are only available at the **server** 服务器端有数据

#### Idea

Federated Matching (FedMatch)

- inter-client consistency loss: aims to maximize the agreement between the models trained at different clients 针对不同的用户设计的Loss函数

$$
\frac{1}{H} \sum_{j=1}^H KL [p^{*}_{\theta^{h_j}}(y|u)||p_{\theta^l(y|u)}]
$$

- parameter decomposition for disjoint learning: **decomposes** the parameters into one for labeled data and the other for unlabeled data for preservation of reliable knowledge, reduction of communication costs, and disjoint learning 针对平衡对标记数据和非标记数据的记忆能力，把模型参数解构为了两部分

$$
minimize \mathcal{L}_u (\psi) = \lambda _{ICCS}\Phi_\sigma^*+\psi(\cdot)+\lambda_{L_2}||\sigma^*-\psi||^2_2+\lambda_{L_1}||\psi||_1
$$

### PERSONALIZED FEDERATED LEARNING WITH FIRST ORDER MODEL OPTIMIZATION



