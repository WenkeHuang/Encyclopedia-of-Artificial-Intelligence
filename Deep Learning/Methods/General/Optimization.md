**Batch Gradient Descent**
$$
\theta_{t+1} = \theta_t -\alpha_t \cdot \frac{1}{n} \cdot \sum_{i=1}^n\nabla_{\theta} J_i(\theta,x^i,y^i)
$$
**Stochastic Gradient Descent**
$$
w_{t+1} = w_{t} - \eta\hat{\nabla}_{w}{L(w_{t})}
$$


**Mnist batch stochastic gradient descent**
$$
\theta_{t+1} = \theta_i -\alpha \frac{1}{m} \cdot \sum_{i=x}^{i=x+m-1}\nabla_{\theta} J_i(\theta,x^i,y^i)
$$
**SGD with Momentum**

A typical value for $\gamma$ is 0.9
$$
v_{t} = \gamma{v}_{t-1} + \eta\nabla_{\theta}J\left(\theta\right) \\
\theta_t = \theta_{t-1} - v_t
$$
**Nesterov Accelerated Gradient**
$$
v_{t} = \gamma{v}_{t-1} + \eta\nabla_{\theta}J\left(\theta-\gamma{v_{t-1}}\right)  \\
 \theta_{t} = \theta_{t-1} + v_{t}
$$
**Polyak Averaging**

Set final parameters to an average of (recent) parameters visited in the optimization trajectory.
$$
\theta_t =\frac{1}{t}\sum_{i}\theta_{i}
$$
**Natural Gradient descent**

NGD is an approximate second-order optimisation method.
$$
g = \frac{\delta{f}\left(z\right)}{\delta{z}} \\
\Delta{z} = \alpha{F}^{−1}g \\
 F = \mathbb{E}_{p\left(t\mid{z}\right)}\left[\nabla\ln{p}\left(t\mid{z}\right)\nabla\ln{p}\left(t\mid{z}\right)^{T}\right]
$$
**Alternating Direction Method of Multipliers**

ADMM problem from (with f , g convex)
$$
minimize f(x)+g(z) \\
st: Ax + Bz = c
$$
two sets fo variables, with separable objective
$$
L_p(x,z,y) = f(x)+g(z)+y^T(Ax+Bz-c)+(\rho /2)||Ax+Bz-c||_2^2
$$
ADMM:
$$
x^{k+1} :=argmin_x L_\rho(x,z^k,y^k) //x-minimization \\
z^{k+1} :=argmin_zL_\rho(x^{k+1},z,y^k) \\
y^{k+1}:=y^k+\rho(Ax^{k+1}+Bz^{k+1}-c)
$$
**Random search**

Random search (RS) is a family of numerical optimization methods that do not require the gradient of the problem to be optimized, and RS can hence be used on functions that are not continuous or differentiable.

**Gradient Clipping**

Handling gradient explosion problem

1. Based on gradient value to clip
2. $L_2$ normalization

$$
\text{ if } ||\textbf{g}||  > v \text{ then } \textbf{g} \leftarrow \frac{\textbf{g}^{v}}{||\textbf{g}||}
$$

**Harris Hawks optimization**

HHO的主要灵感来源于Harris的鹰在自然界中的合作行为和追逐方式，称之为突然袭击。在这种智能策略中，几只老鹰合作从不同的方向扑向猎物，试图给它一个惊喜。

HHO is a popular swarm-based, gradient-free optimization algorithm with several active and time-varying phases of exploration and exploitation.

**Hybrid Firefly and Particle Swarm Optimization**

HFPSO is a metaheuristic optimization algorithm that combines strong points of firefly and particle swarm optimization. HFPSO tries to determine the start of the local search process properly by checking the previous global best fitness values.

**AdaGrad**
$$
 \theta_{t+1, i} = \theta_{t, i} - \frac{\eta}{\sqrt{G_{t, ii} + \epsilon}}g_{t, i} 
$$
The benefit of AdaGrad is that it eliminates the need to manually tune the learning rate; most leave it at a default value of 0.01. Its main weakness is the accumulation of the squared gradients in the denominator.

**AdaDelta**
$$
E\left[g^{2}\right]_{t} = \gamma{E}\left[g^{2}\right]_{t-1} + \left(1-\gamma\right)g^{2}_{t}
$$
Usually $\gamma$ is set to around $0.9$. 

AdaDelta takes the form:
$$
 \Delta\theta_{t} = -\frac{\eta}{\sqrt{E\left[g^{2}\right]_{t} + \epsilon}}g_{t} 
$$
**RMSProp**

**RMSProp** is an unpublished adaptive learning rate optimizer proposed by Geoff Hinton. The motivation is that the magnitude of gradients can differ for different weights, and can change during learning, making it hard to choose a single global learning rate. 

**Adaptive moment estimation (Adam)**

Adam is an adaptive learning rate optimization algorithm that utilises both momentum and scaling, combining the benefits of RMSProp and SGD w/th Momentum. The optimizer is designed to be appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients.

The weight updates are performed as:
$$
w_{t} = w_{t-1} - \eta\frac{\hat{m}_{t}}{\sqrt{\hat{v}_{t}} + \epsilon}
$$
with
$$
\hat{m}_{t} = \frac{m_{t}}{1-\beta^{t}_{1}}\\
 \hat{v}_{t} = \frac{v_{t}}{1-\beta^{t}_{2}} \\
 m_t = \beta_1 m_{t-1}+ (1-\beta_1)g_t \\
 v_t = \beta_2 v_{t-1}+(1-\beta_2)g_t^2
$$
$\eta$ is the step size / learning rate, around $1e-3$ in the original paper. $ \epsilon $is a small number, typically $1e-8$ or $1e-10$, to prevent dividing by zero.  $\beta_1$ and $\beta_2$ are forgetting parameters, with typical values $0.9$ and $0.999$

**Adamax**

Adamax是Adam的一种变体，此方法对学习率的上限提供了一个更简单的范围。公式上的变化如下：
$$
u_{t} = \beta^{\infty}_{2}v_{t-1} + \left(1-\beta^{\infty}_{2}\right)|g_{t}|^{\infty}\\
=max(\beta_2 \cdot v_{t-1},|g_t|)
$$
We can plug into the Adam update equation by replacing $\sqrt{\hat{v}_{t} + \epsilon}$ with $u_t$ to obtain the AdaMax update rule:
$$
\theta_{t+1} = \theta_{t} - \frac{\eta}{u_{t}}\hat{m}_{t}
$$
**Nesterov-accelerated Adaptive Moment(NADAM)**

Estimation, combines Adam and Nesterov Momentum. The update rule is of the form:
$$
\theta_{t+1} = \theta_{t} - \frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon}\left(\beta_{1}\hat{m}_{t} + \frac{(1-\beta_{t})g_{t}}{1-\beta^{t}_{1}}\right)
$$
**Online Hard Example Mining**

Some object detection datasets contain an overwhelming number of easy examples and a small number of hard examples. Automatic selection of these hard examples can make training more effective and efficient. OHEM, or Online Hard Example Mining, is a bootstrapping technique that modifies SGD to sample from examples in a non-uniform way depending on the current loss of each example under consideration. The method takes advantage of detection-specific problem structure in which each SGD mini-batch consists of only one or two images, but thousands of candidate examples. The candidate examples are subsampled according to a distribution that favors diverse, high loss instances.

**Two Time-scale Update Rule**

The Two Time-scale Update Rule (TTUR) is an update rule for generative adversarial networks trained with stochastic gradient descent. TTUR has an individual learning rate for both the discriminator and the generator. The main premise is that the discriminator converges to a local minimum when the generator is fixed. If the generator changes slowly enough, then the discriminator still converges, since the generator perturbations are small. Besides ensuring convergence, the performance may also improve since the discriminator must first learn new patterns before they are transferred to the generator. In contrast, a generator which is overly fast, drives the discriminator steadily into new regions without capturing its gathered information.

**AdaW**

AdamW is a stochastic optimization method that modifies the typical implementation of weight decay in Adam, by decoupling weight decay from the gradient update. To see this, $L_2$ regularization in Adam is usually implemented with the below modification where $w_t$ is the rate of the weight decay at time $t$:
$$
 g_{t} = \nabla{f\left(\theta_{t}\right)} + w_{t}\theta_{t}
$$
while AdamW adjusts the weight decay term to appear in the gradient update:
$$
\theta_{t+1, i} = \theta_{t, i} - \eta\left(\frac{1}{\sqrt{\hat{v}_{t} + \epsilon}}\cdot{\hat{m}_{t}} + w_{t, i}\theta_{t, i}\right), \forall{t}
$$
**LARS**

Layer-wise Adaptive Rate Scaling, or LARS, is a large batch optimization technique. There are two notable differences between LARS and other adaptive algorithms such as Adam or RMSProp: 

- first, LARS uses a separate learning rate for each layer and not for each weight. 
- And second, the magnitude of the update is controlled with respect to the weight norm for better control of training speed.

$$
m_{t} = \beta_{1}m_{t-1} + \left(1-\beta_{1}\right)\left(g_{t} + \lambda{x_{t}}\right)
$$

$$
x_{t+1}^{\left(i\right)} = x_{t}^{\left(i\right)}  - \eta_{t}\frac{\phi\left(|| x_{t}^{\left(i\right)} ||\right)}{|| m_{t}^{\left(i\right)} || }m_{t}^{\left(i\right)} 
$$

**Adafactor**

Adafactor is a stochastic optimization method based on Adam that reduces memory usage while retaining the empirical benefits of adaptivity. 

This is achieved through maintaining **a factored representation of the squared gradient accumulator** across training steps.

Specifically, by tracking moving averages of the row and column sums of the squared gradients for matrix-valued variables, we are able to reconstruct a low-rank approximation of the exponentially smoothed accumulator at each training step that is optimal with respect to the generalized Kullback-Leibler divergence. 

For an $n \times m$ matrix, this reduce the memory requirements from $O(n m)$ to $O(n+m)$.

**Stochastic Weight Averaging**

Stochastic Weight Averaging is an optimization procedure that averages multiple points along the trajectory of SGD, with **a cyclical or constant learning rate**. On the one hand it averages weights, but it also has the property that, with a cyclical or constant learning rate, SGD proposals are approximately sampling from the loss surface of the network, leading to stochastic weights and helping to discover broader optima.

**LAMB**

LAMB is a a **layerwise adaptive large batch optimization technique**. It provides a strategy for adapting the learning rate in large batch settings. LAMB uses Adam as the base algorithm and then forms an update as:
$$
r_{t} = \frac{m_{t}}{\sqrt{v_{t}} + \epsilon}
$$

$$
x_{t+1}^{\left(i\right)} = x_{t}^{\left(i\right)}  - \eta_{t}\frac{\phi\left(|| x_{t}^{\left(i\right)} ||\right)}{|| m_{t}^{\left(i\right)} || }\left(r_{t}^{\left(i\right)}+\lambda{x_{t}^{\left(i\right)}}\right) 
$$

**AMSGrad**

AMSGrad is a stochastic optimization method that seeks to fix a convergence issue with Adam based optimizers. AMSGrad uses the maximum of past squared gradients $v_t$ rather than the exponential average to update the parameters:
$$
m_{t} = \beta_{1}m_{t-1} + \left(1-\beta_{1}\right)g_{t} \\
v_{t} = \beta_{2}v_{t-1} + \left(1-\beta_{2}\right)g_{t}^{2} \\
 \hat{v}_{t} = \max\left(\hat{v}_{t-1}, v_{t}\right) \\
 \theta_{t+1} = \theta_{t} - \frac{\eta}{\sqrt{\hat{v}_{t}} + \epsilon}m_{t}
$$




















