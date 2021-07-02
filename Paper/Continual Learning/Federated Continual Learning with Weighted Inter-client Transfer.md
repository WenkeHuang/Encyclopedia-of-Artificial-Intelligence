# Federated Continual Learning with Weighted Inter-client Transfer

## Problem

- Yet little research has been done regarding the scenario where each client learns on a sequence of tasks from private local data stream. （少有研究在乎当每个clients 学习一系列的任务从私有的任务流）
- Federated Continual Learning poses new challenges to continual learning：such as utilizing knowledge from other clients, while preventing interference from irrelevant knowledge. 
  - 不仅continual learning存在灾难性遗忘，也存在其他客户端可能的推断
  - 正向迁移促进模型性能，负向迁移降低模型性能，因此需要选择性的利用其他客户端的知识来minimize inter-client interference 并且 maximize inter-client knolwedge transfer
  - efficient communication，当利用其他客户端知识会导致急哦啊刘成本变大，因而我们希望知识to be represented as compactly as possible

## Existing Methods

## Motivation

Motivated by the human learning process from indirect expe- riences, we introduce a novel continual learning under fed- erated learning setting, which we refer to as *Federated Continual Learning (FCL)*

FCL 假设多方用户在私有数据集流上进行一系列的任务，并且通过全局服务器交流学习的参数

### Problem Definition

Lean from a sequence of tasks $\{\mathcal{T}^{1},\mathcal{T}^{2},...\mathcal{T}^{T}\}$, where $\mathcal{T}^{t}$ is a labeled dataset of $t^{th}$ task, $\mathcal{T}^{t}=\{x_i^{(t)},y_i^{(t)}\}_{i=1}^{N_t})$ and minimize 
$$
\theta^{(t)}\mathcal{L}(\theta^{(t)};\theta^{(t-1)},\mathcal{T}^t)
$$
$\theta^{(t)} \in \mathbb{R}^{N \times M}$ is a set of the parameters in the model at taask t

**As for federated learning setting**

we have $C$ clients, where at each client $c \in \{c_1,...,c_C\}$ trains a model on a privately acces- sible sequence of tasks $\{\mathcal{T}^{(1)}_c,\mathcal{T}^{2}_c,...,\mathcal{T}^{t}_c \subseteq \mathcal{T} \}$

Now the goal is to effectively train $C$ continual learning models on their own private task streams, via communicating the model parameters with the global server, which aggregates the parameters sent from each client, and redistributes them to clients.

## Idea

*Federated Weighted Inter-client Transfer (FedWeIT)*

Decompose the model parameters into a dense global parameter and sparse task-adaptive parameters （解构模型参数为密集全局模型和稀疏任务适应参数）

- Global parameters $\theta_G$ encode task-generic knowledge

- Task-specific knowledge will be encoded into the task-adaptive parameters $A_c^{(t)}$

$$
\theta_c^{(t)}=B_c^{(t)} \odot m_c^{(t)}+A_c^{(t)}+\sum_{i \in C }\sum_{j < |t|}\alpha_{i,j}^{(t)}A_i^{(j)}
$$

$B_c^{(t)} \in \mathbb{R}^{N \times M}$  is the set of base parameters for $c^{th}$ client shared across all tasks in the client.

$m_c^{(t)} \in \mathbb{R}^M$ is a spase mask which allows to adaptively transform $B_c^{(t)}$ for the target task.

$A_c^{(t)} \in \mathbb{R}^{N \times M}$  is a sparse matrix oft ask-adaptive parameters for the task $t$ at client $c$.

**Explanation**

$B_c^{(t)}$

The first term allows selective utilization of the global knowl- edge. We want the base parameter $B_c^{(t)}$ at each client to capture generic knowledge across all tasks across all clients.

$m_c^{(t)}$

we learn the sparse mask $m_c^{(t)}$ to select only the relevant parameters for the given task. This sparse parameter selection helps minimize inter-client interference, and also allows for efficient communication.

$A_c^{(t)}$

Since we additively decompose the parameters, this will learn to capture knowledge about the task that is not captured by the first term, and thus will capture specific knowledge about the task $\mathcal{T}^{(t)}_c$

$\sum_{i \in C }\sum_{j < |t|}\alpha_{i,j}^{(t)}A_i^{(j)}$

The final term describes inter-client knowledge transfer. **We have a set of parameters that are transmitted from the server, which contain all task-adaptive parameters from all the clients**. To selectively utilizes these indirect experiences from other clients, we further allocate attention $\alpha_c^{t}$ on these parameters, to take a weighted combination of them.

**Training**
$$
\min_{B_c^{(t)},m_c^{(t)},A_c^{(1:t)},\alpha_c^{(t)}} \mathcal{L}(\theta_c^{(\theta_c^{(t)};\mathcal{T}_c)}+\lambda_1 \Omega\{m_c^{(t),A_c^{(1:t)}}\})+\lambda_2 \sum_{i=1}^{t-1}||\Delta B_c^{(t)} \odot m_c^{(i)}-\Delta A_c^{(i)}||^2_2
$$
$\mathcal{L}$

Loss function

$\Omega (\cdot)$

A sparsity-inducing regularization term for the task adaptive parameter and the masking variable (we use $\mathcal{l_1}$-norm regularization) **to make them sparse**

$\sum||||^2_2$

The final regularization term is used for retroac- tive update of the past task-adaptive parameters, which helps the task-adaptive parameters to maintain the original solu- tions for the target tasks, by **reflecting the change of the base parameter**.

## Conclusion

我们解决了联邦持续学习的一个新问题，其目标是在每个客户机上持续学习本地模型，同时允许它利用来自其他客户机的间接经验（任务知识）。这带来了新的挑战，如客户间的知识转移和防止客户间的干扰无关的任务。为了应对这些挑战，我们将每个客户机上的模型参数额外分解为在所有客户机上共享的全局参数和特定于每个任务的稀疏局部任务自适应参数。此外，我们允许每个模型有选择地更新全局任务共享参数，并有选择地利用来自其他客户机的任务自适应参数。在不同任务相似性下，针对已有的联邦学习和连续学习基线，对我们的模型进行了实验验证，结果表明，我们的模型在降低通信开销的前提下取得了显著的性能。我们相信，联合持续学习是一个实践上的重要课题，对持续学习和联合学习的研究界都有很大的兴趣，这将导致新的研究方向。

