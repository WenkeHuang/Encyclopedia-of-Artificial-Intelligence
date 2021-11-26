# Continual Learning Through Synaptic Intelligence

## Problem

**Catastrophic forgetting**

- Non-stationary data 
- New memories overwrite old ones 
- Capacity not the issue

## Motivation

Study the role of internal synaptic dynamics to enable ANNs to learn sequences of classification tasks.

To tackle the problem of continual learning in neural net- works, we sought to build a simple structural regularizer that could be computed online and implemented locally at each synapse. 

## Related Work

- EWC

☑️ More influential parameters are pulled back more strongly towards a reference weight with which good performance was achieved on previous tasks.

❌ EWC relies on a point estimate of the diagonal of the Fisher info mation metric at the final parameter values, which has to be computed during a separate phase at the end of each task.

*Here, we are putting forward a method which computes an importance measure online and along the entire learning trajectory,*

## Idea

To that end, we developed a class of algorithms which keep track of an importance measure $w_k^{\mu}$ which reflects past credit for improvements of the task objective $L_{\mu}$ for task $\mu$ to individual synapses $\theta_k$.
$$
L(\theta(t)+\delta(t))-L(\theta(t)) \approx \sum_i g_i(t)\delta_i(t)
$$

$$
g_i(t) \approx \frac{\partial L}{\partial \theta_i}
$$

$$
\delta_i(t)=\theta_i^{'}(t)=\frac{\partial{\theta_i}}{\partial t}
$$

$$
\int_{t^{\mu-1}}^{t^\mu} g(t)\delta(t) dt = \sum_i \int_{t^{\mu-1}}^{t^\mu} g_i(t)\delta_i(t) dt = - \sum_i w_i^\mu
$$

$w_i^\mu$ 就是参数$\theta_i$ 的变化对损失函数输出的影响。在 offline 场景下，$w_i^\mu$  直接就能通过损失函数输出值的变化量算出来。与 EWC 和 MAS 不同的是，IS 还可以在 online 场景下计算$w_i^\mu$ ，这时 ![[公式]](https://www.zhihu.com/equation?tex=g_i%28t%29) 可以用 $g_i(t)=\frac{\partial{L}}{\partial{\theta_i}}$ 来近似，而 $\delta_i(t)$ 就相当于 $\delta_i(t)=\theta_i^{'}=\frac{\partial \theta_i}{\partial t}$。
$$
\Delta_i^v = \theta_i (t^v)-\theta_i(t^{v-1})
$$

$$
\Omega_k^\mu = \sum_{v < \mu} \frac{w_k^v}{(\Delta_k^v)^2+\xi} （参数变化对损失函数的影响，下面是参数值的变化）
$$

Note that the term in the denominator $\Delta_k^v$ ensures that the regularization term carries the same units as the loss L. For practical reasons we also introduce an additional damping parameter $\xi$ to bound the expression in cases where $\Delta_k^v \rightarrow 0$


$$
\widetilde{L} = L_\mu + c\sum_k \Omega_k^\mu (\widetilde{\theta}_k-\theta_k)^2
$$

## Theoretical analysis of special cases

Geometry of a simple quadratic error function
$$
E(\theta)=\frac{1}{2}(\theta-\theta^*)^TH(\theta-\theta^*)
$$
进一步考虑这个误差函数的批梯度下降动力学。在小的离散时间学习率的限制下，这种下降动力学用连续时间微分方程来描述:
$$
\tau \frac{d \theta}{d t}= - \frac{\partial E}{\partial \theta}= -H (\theta - \theta^*)
$$

$$
\theta(t)=\theta^*+e^{-H \frac{t}{\tau}}(\theta(0)-\theta^*)
$$

而梯度下降又遵守：
$$
g = \tau \frac{d \theta}{d t}
$$
Q

## Conclusion

- Individual synapses can estimate their importance as contribution to changes in loss 

- They can do this on-line by efficiently computing the path integral over the entire parameter trajectory 
- Exploiting this information intelligently 
  - Alleviates catastrophic forgetting 
  - Yields better generalization

