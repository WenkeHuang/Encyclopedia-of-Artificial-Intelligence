# Explantion of EWC

## Likelihood Function & Probability Function

对于这个函数： $p(x|\theta)$ 输入有两个：x表示某一个具体的数据； $\theta$ 表示模型的参数

如果 $\theta$  是已知确定的， 是x变量，这个函数叫做概率函数(probability function)，它描述对于不同的样本点  x，其出现概率是多少

如果 x 是已知确定的， $\theta$  是变量，这个函数叫做似然函数(likelihood function), 它描述对于不同的模型参数，出现  x 这个样本点的概率是多少

## Maximum Likelihoood Estimation & Bayesian Estimation

**解决问题的本事就是求$\theta$**

1. **频率学派**

存在唯一真值 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta+) 。举一个简单直观的例子--抛硬币，我们用 ![[公式]](https://www.zhihu.com/equation?tex=P%28head%29) 来表示硬币的bias。抛一枚硬币100次，有20次正面朝上，要估计抛硬币正面朝上的bias ![[公式]](https://www.zhihu.com/equation?tex=P%28head%29%3D%5Ctheta) 。在频率学派来看，![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) = 20 / 100 = 0.2，很直观。当数据量趋于无穷时，这种方法能给出精准的估计；然而缺乏数据时则可能产生严重的偏差。例如，对于一枚均匀硬币，即 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) = 0.5，抛掷5次，出现5次正面 (这种情况出现的概率是1/2^5=3.125%)，频率学派会直接估计这枚硬币 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) = 1，出现严重错误。

2. **贝叶斯学派**

 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 是一个随机变量，符合一定的概率分布。在贝叶斯学派里有两大输入和一大输出，输入是先验 (prior)和似然 (likelihood)，输出是后验 (posterior)。*先验*，即 ![[公式]](https://www.zhihu.com/equation?tex=P%28%5Ctheta%29) ，指的是在没有观测到任何数据时对 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 的预先判断，例如给我一个硬币，一种可行的先验是认为这个硬币有很大的概率是均匀的，有较小的概率是是不均匀的；*似然*，即 ![[公式]](https://www.zhihu.com/equation?tex=P%28X%7C%5Ctheta%29) ，是假设 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 已知后我们观察到的数据应该是什么样子的；*后验*，即 ![[公式]](https://www.zhihu.com/equation?tex=P%28%5Ctheta%7CX%29) ，是最终的参数分布。贝叶斯估计的基础是贝叶斯公式，如下：
$$
P(\theta|X)=\frac{P(X|\theta) \times P(\theta)}{P(X)}
$$
这里有两点值得注意的地方：

- 随着数据量的增加，参数分布会越来越向数据靠拢，先验的影响力会越来越小
- 如果先验是uniform distribution，则贝叶斯方法等价于频率方法。因为直观上来讲，先验是uniform distribution本质上表示对事物没有任何预判

**MLE**

Maximum Likelihood Estimation, MLE是频率学派常用的估计方法！

假设数据 ![[公式]](https://www.zhihu.com/equation?tex=x_1%2C+x_2%2C+...%2C+x_n+) 是i.i.d.的一组抽样，![[公式]](https://www.zhihu.com/equation?tex=X+%3D+%28x_1%2C+x_2%2C+...%2C+x_n%29) 。其中i.i.d.表示Independent and identical distribution，独立同分布。那么MLE对 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 的估计方法可以如下推导：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%2A%7D+%5Chat%7B%5Ctheta%7D_%5Ctext%7BMLE%7D+%26%3D+%5Carg+%5Cmax+P%28X%3B+%5Ctheta%29+%5C%5C+%26%3D+%5Carg+%5Cmax+P%28x_1%3B+%5Ctheta%29+P%28x_2%3B+%5Ctheta%29+%5Ccdot%5Ccdot%5Ccdot%5Ccdot+P%28x_n%3B%5Ctheta%29+%5C%5C+%26+%3D+%5Carg+%5Cmax%5Clog+%5Cprod_%7Bi%3D1%7D%5E%7Bn%7D+P%28x_i%3B+%5Ctheta%29+%5C%5C+%26%3D+%5Carg+%5Cmax+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5Clog+P%28x_i%3B+%5Ctheta%29+%5C%5C+%26%3D+%5Carg+%5Cmin+-+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5Clog+P%28x_i%3B+%5Ctheta%29+%5Cend%7Balign%2A%7D)



**Bayes's Rule**
$$
p(\theta|D)=\frac{p(D|\theta) \cdot p(\theta)}{p(D)}
$$

$$
log \ p(\theta|D)= log \ p(D|\theta) + log \ p(\theta)-log \ p(D)
$$


Hence
$$
log \ p(\theta|D)= log \ (p(D_B|\theta) + log \ p(\theta|D_A)-log \ p(D_B)
$$
Note that the left hand side is still describing the posterior probability of the parameters given the entire dataset, while the right hand side only depends on the loss function for task B $log \ p(D_B|\theta)$

$p(\theta|A)$ : This posterior probability must contain information about **which parameters were important to task A and is therefore the key to implementing EWC.**
$$
\mathcal{L}(\theta) = \mathcal{L}_B(\theta)+\sum_i \frac{\lambda}{2}F_i(\theta_i-\theta^*_{A,i})^2
$$
$\mathcal{L}_B(\theta)$ is the loss for task B only, $\lambda$ sets how important the old task is compared to the new one and $i$ labels each parameter

## Reference

聊一聊机器学习的MLE和MAP：最大似然估计和最大后验估计 - 夏飞的文章 - 知乎 https://zhuanlan.zhihu.com/p/32480810







=
