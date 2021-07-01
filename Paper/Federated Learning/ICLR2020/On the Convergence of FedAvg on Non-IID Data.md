# On the Convergence of FedAvg on Non-IID Data

## Problems

FL has **three** unique characters that distinguish it from the standard parallel optimization

- First, the training data are massively distributed over an incredibly large number of devices, and the connection between the central server and a device is slow.  

  训练数据很多并且分布在很大规模的蛇别上，中央服务器和设备之间的连接的慢的。

- Second, unlike the traditional distributed learning systems, the FL system does not have control over users’ devices. 

  其次，相比于传统的分布式学习，联邦学习系统不能控制用户设备

- Third, the training data are non-iid, that is, a device’s local data cannot be regarded as samples drawn from the overall distribution.

  训练数据也是非独立同分布的，一个用户的本地数据不能被看作是从整体分布中采样出来的

## Related Work

There have been much efforts developing convergence guarantees for FL algorithm based on the assumptions that 

1. **the data are iid** 
2. **all the devices are active.**

## Notation

$N$ : The total number of user devices 

$K (\leq N)$  :The maximal number of devices that participate in every round’s communication

$ T$ : The total number of every device’s SGDs

$E$ : The number of local iterations performed in a device between two communications

$\frac{T}{E}$ : The number of communications

## FedAVG

Distributed optimization model
$$
\min_w \{F(w)\triangleq \sum_{k=1}^N p_kF_k(w)\}
$$
Local objective
$$
F_k(w)\triangleq \frac{1}{n_k}\sum_{j=1}^{n_k} \mathcal{l}(w;x_{k,j})
$$
**E local update**
$$
w_{t+i+1}^k \leftarrow w_{t+i}^k - \eta_{t+i}\nabla F_k(w_{t+i}^k,\xi_{t+i}^k),i=0,1,..,E-1
$$
$\eta_{t+i}$ : is the learning rate

$\xi_{t+i}^k$ is a sample uniformly chosen from the $k-th$ device's local data uniformly at random

**Full device participation**
$$
w_{t+E}\leftarrow \sum_{k=1}^N p_k \ w_{t+E}^k
$$
**Partial device participation**
$$
w_{t+E}\leftarrow \frac{N}{K} \sum_{k \in S_t} p_k \ w_{t+E}^k
$$

## Convergence Analysis of FedAvg in Non-IID Setting

**Assumption**

- $F_1,...,F_N$ are all $L-smooth$: for all $v$ and $w$, $F_k(v)\leq F_k(w)+(v-w)^T \triangledown F_k(w)+\frac{L}{2}||v-w||^2_2$
- $F_1,...,F_N$ are all $\mu-strongly convex$: for all $v$ and $w$,  $F_k(v)\geq F_k(w)+(v-w)^T \triangledown F_k(w)+\frac{\mu}{2}||v-w||^2_2$ 
- The variance of stochastic gradients in each device is bounded $\mathbb{E} || \triangledown F_k(w_t^k,\xi_t^k)||^2 \leq G^2$ for all $k=1,...,N$
- The expected squared norm of stochastic gradients is uniformly bounded $\mathbb{E} \triangledown F_k(w_t^k,\xi_t^k)||^2 \leq G^2$   for all $k=1,...,N$ and $t=1,...,T-1$

**Quantifying the degree of non-iid (heterogeneity)**

$F^*$ and $F^*_k$ be the minimum values of $F$ and $F_k$, $\Gamma = F^* - \sum_{k=1}^N p_kF^*_k$ for quantifying the degree of non-iid.

If the data are iid, then $\Gamma$ obviously goes to zero as the number of sample  grows.

If the data are non-iid, then $\Gamma$ is nonzero, and its magnitude reflects the heterogeneity of the data distribution.

**Convergence Result: Full Device Participation**

$\kappa=\frac{L}{\mu}$ 

$\gamma = \max\{8\kappa,E\}$ 

$\eta_t = \frac{1}{\mu(\gamma +t)}$
$$
\mathbb{E}[F(w_T)]-F^* \leq \frac{\kappa}{\gamma + T -1 }\frac{2B}{\mu}+\frac{\mu \gamma}{2}\mathbb{E}||w_1-w^*||^2
$$
Where 
$$
B = \sum_{k=1}^N p_k^2 \sigma^2_k + 6L \Gamma+8(E-1)^2G^2
$$
**Convergence Result: Partial Device Participation**

$C = \frac{4}{k}E^2G^2$
$$
\mathbb{E}[F(w_T)]-F^* \leq \frac{k}{\gamma + T -1 }\frac{2(B+C)}{\mu}+\frac{\mu \gamma}{2}\mathbb{E}||w_1-w^*||^2
$$


## Contribution

**The number of communication:**
$$
\frac{T}{E} = \mathcal{O}[\frac{1}{\epsilon}((1+\frac{1}{K})EG^2+\frac{\sum_{k=1}^Np_k^2\sigma^2_k+\Gamma+G^2}{E}+G^2)]
$$
$E$ is a **knob controlling** the convergence rate: neither setting $E$ over-small ($E = 1$  makes $FedAvg$ equivalent to SGD) nor setting $E$ over-large is good for the convergence.

联邦学习在机器学习和优化社区中越来越流行。本文研究了适用于联邦环境的启发式算法FedAvg的收敛性。我们研究了**采样和平均方案**的影响。我们为两种方案提供了理论保证，并对其性能进行了实证分析。我们的工作有助于对FedAvg的理论理解，并为实际应用中的算法设计提供启示。虽然我们的分析局限于凸优化问题，但我们希望我们的见解和证明技术能对今后的工作有所启发。

































