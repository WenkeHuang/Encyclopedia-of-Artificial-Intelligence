# Overcoming forgetting in federated learning on non-IID data

## Background

**Three challenges in Federated Learning**

1. The number of computing stations 可能会达到hundreds of millions

2. much slower 的交流 相比于inter cluster communication found in data centers

3. 高度的非i.i.d 的数据分布方式

## Related Work

One approach is to just give up the periodical averaging, and reduce the communication by sparsification and quantization of the updates sent to the central point after each local mini batch. （我们的方法 放弃了 周期性的平均策略，并且降低了通信损失，在每个本地mini batch后，发送更新到中央服务器通过稀疏和量化的方式）

MOCHA 算法使用了primal-dual formulation 为了解决优化问题，这是不适合深度网络的

## Motivation

*Elastic Weight Consolidation(EWC)*

EWC的目的是防止从学习任务A转移到学习任务B时造成灾难性的遗忘。其想法是识别网络参数θ中对任务A最有帮助的坐标，当taskB学习的时候，**进行惩罚限制更新参数**

In order to control the stiffness of $\theta$ per coordinate while learning task $B$, the authors suggest to use the diagonal of the **Fisher information matrix** $\mathcal{I}_A^*=\mathcal{I}_A(\theta_A^*)$ to selectively penalize parts of the
parameters vector $\theta$ that are getting too far from $\theta_A^*$
$$
\widetilde{L}(\theta) = L_B(\theta)+\lambda(\theta-\theta_A^*)^Tdiag(\mathcal{I_A^*})(\theta-\theta_A^*)
$$
The formal justification they provide for it is Bayesian: Let $D_A$ and $D_B$ be independent datasets used for tasks A and B. We have that：
$$
log \ p(\theta|D_A \ and \ D_B) = log \ p(D_B | \theta)+log \ p(\theta|D_A)-log \ p(D_B)
$$
$log \ p(D_B | \theta)$ is just the standard likelihood maximized in the optimization of $L_B(\theta)$

the posterior $p(\theta|D_A)$ is approximated with Laplace’s method as a Gaussian distribution with expectation $\theta_A^*$ and convariance diag($\mathcal{I}_A^*$)
$$
\widetilde{L}(\theta) \approx L_B(\theta)+ \frac{1}{2}(\theta-\theta_A^*)^T H_{L_A}(\theta-\theta_A^*) \approx L_B(\theta)+L_A(\theta)
$$

## Federated Curvature

$$
\widetilde{L}_{t,s}(\theta) = L_s(\theta)+\lambda \sum_{j \in S \setminus s}(\theta-\widehat{\theta}_{t-1,j})^T diag(\widehat{\mathcal{I}}_{t-1,j})(\theta-\widehat{\theta}_{t-1,j})
$$

**On each round t**

$\widehat{\theta}_t=\frac{1}{N}\sum_{i=1}^N \widehat{\theta}_{t-1,j}$

the nodes optimize their local loss by running SGD for E local epoch

**At the end of each round t**, each node **j** sends to the rest of the models the SGD result $\widehat{\theta}_{t,j}$ and $diag(\widehat{I}_{t,j})$ will be used for the loss of round $t+1$.

### Keeping Low Bandwidth and Preserving Privacy

$$
\widetilde{L}_{t,s}(\theta) = L_s(\theta)+\lambda \theta^T[\sum_{j \in S\setminus s }diag(\widehat{I}_{t-1,j})]\theta-2\lambda\theta^T\sum_{j \in S \setminus } diag (\widehat{I}_{t-1,j}\widehat{\theta}_{t-1,j})+const
$$

**Bandwidth** The central point needs only to maintain and transmit to the edge node two additional elements, besides $\theta$, of the same size as $\theta$,
$$
u_t = \sum_{j \in S}diag(\widehat{I}_{t-1,j})
$$

$$
v_t = \sum_{j \in S} diag (\widehat{I}_{t-1,j})\widehat{\theta}_{t-1,j}
$$

## Fisher Information

$$
L(X;\theta) = \prod_{i=1}^n f(X_i;\theta)
$$

为了解得Maximum Likelihood Estimate(MLE)，我们要让log likelihood的一阶导数得0，然后解这个方程，得到$\widehat{\theta}_{MLE}$
$$
S(X;\theta) = \sum_{i=1}^n \frac{\partial logf(X_i;\theta)}{\partial \theta}
$$
那么Fisher Information，用$\mathcal{I}(\theta)$表示，的定义就是这个Score function的二阶矩（second moment）$I(\theta)=E[S(X;\theta)^2]$

**Fisher Information的第一条数学意义：就是用来估计MLE的方程的方差**

它的直观表述就是，随着收集的数据越来越多，这个方差由于是一个Independent sum的形式，也就变的越来越大，也就象征着得到的信息越来越多。

**Fisher Information的第二条数学意义：log likelihood在参数真实值处的负二阶导数的期望**
$$
E[S(X;\theta)^2]=-E[\frac{\partial^2}{\partial\theta^2}log:(X;\theta)]
$$
对于这样的一个log likelihood function，它越平而宽，就代表我们对于参数估计的能力越差，它高而窄，就代表我们对于参数估计的能力越好，也就是信息量越大。而这个log likelihood在参数真实值处的负二阶导数，就反应了这个log likelihood在顶点处的弯曲程度，弯曲程度越大，整个log likelihood的形状就越偏向于高而窄，也就代表掌握的信息越多。



























































