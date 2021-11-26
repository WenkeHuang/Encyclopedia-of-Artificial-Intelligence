# Explantion of EWC

## Likelihood Function & Probability Function

一文搞懂极大似然估计 - 忆臻的文章 - 知乎 https://zhuanlan.zhihu.com/p/26614750

## Maximum Likelihoood Estimation & Bayesian Estimation

聊一聊机器学习的MLE和MAP：最大似然估计和最大后验估计 - 夏飞的文章 - 知乎 https://zhuanlan.zhihu.com/p/32480810

## Gradient & Jacobi Matrix & Hessian Matrix

Let $f: \mathbb{R}^n \rightarrow \mathbb{R}$ be a scalar field. The gradient, $\nabla f: \mathbb{R}^n \rightarrow \mathbb{R}^n$ is a vector, such that $(\nabla f)_j = \partial f/ \partial x_j$. Because every point in $\text{dom}(f)$ is mapped to a vector, then $\nabla f$ is a vector field.

https://math.stackexchange.com/questions/3680708/what-is-the-difference-between-the-jacobian-hessian-and-the-gradient

- Gradient: Vector of first order derivatives of a scalar field.
- Jacobian: Matrix of gradients for components of a vector field.
- Hessian: Matrix of second order mixed partials of a scalar field.

## Taylor's Formula

$$
\begin{align*}
   f(x) \approx f(a) + Df(a) (x-a)
   +  \frac{1}{2} (x-a)^T Hf(a) (x-a).
\end{align*}
$$

雅可比矩阵、黑森矩阵、泰勒展开式 - 致Great的文章 - 知乎 https://zhuanlan.zhihu.com/p/90496291

基于Hessian矩阵，就可以判断多元函数的极值情况，结论如下：

- 如果是正定矩阵，则临界点处是一个局部极小值
- 如果是负定矩阵，则临界点处是一个局部极大值
- 如果是不定矩阵，则临界点处不是极值

## Leibniz integral rule

https://en.wikipedia.org/wiki/Leibniz_integral_rule
$$
\frac{d}{dx}(\int_{a(x)}^{b(x)} f(x,t) dt)=\int_{a(x)}^{b(x)} \frac{\partial }{\partial x}f(x,t) dt +f( x, b(x)) \frac{db}{dx}-f( x, a(x)) \frac{da}{dx}
$$
If $a(x)$ and $b(x)$ are constants (i.e., $a\neq a(x)$ and $b\neq b(x)$, then $\frac{da}{dx}=0$ and $\frac{db}{dx}=0$and hence the first theorem becomes:
$$
\frac{d}{dx}(\int_{a}^{b} f(x,t) dt)=\int_{a}^{b} \frac{\partial }{\partial x}f(x,t) dt
$$
https://math.stackexchange.com/questions/3231347/leibniz-integral-rule-and-fundamental-thm-of-calculsdifferences

积分符号内取微分是一种什么方法？ - 罗旻杰的回答 - 知乎 https://www.zhihu.com/question/24481887/answer/27948494

积分符号内取微分是一种什么方法？ - 战神阿瑞斯的回答 - 知乎 https://www.zhihu.com/question/24481887/answer/433338589

## Fisher Information & Hessian

https://math.stackexchange.com/questions/265917/intuitive-explanation-of-a-definition-of-the-fisher-information

Now you could see why summarizing uncertainty (curvature) about the likelihood function takes the particular formula of Fisher information.

[Fisher] Information may be seen to be a measure of the "curvature" of the support curve near the maximum likelihood estimate of θ. A "blunt" support curve (one with a shallow maximum) would have a low negative expected second derivative, and thus low information; while a sharp one would have a high negative expected second derivative and thus high information.



To answer an additional question by the OP, I will show what the expectation of the score is zero. Since $p \left( x, \theta \right)$ is a density
$$
\begin{eqnarray*}
  \int p \left( x ; \theta \right) \mathrm{d} x & = & 1
\end{eqnarray*}
$$
Take derivatives on both sides
$$
\begin{eqnarray*}
  \frac{\partial}{\partial \theta} \int p \left( x ; \theta \right) \mathrm{d}
  x & = & 0
\end{eqnarray*}
$$
Looking on the left hand side (**Leibniz integral rule**)
$$
\begin{eqnarray*}
  \frac{\partial}{\partial \theta} \int p \left( x ; \theta \right) \mathrm{d}
  x & = & \int \frac{\partial p \left( x ; \theta \right)}{\partial \theta}
  \mathrm{d} x\\
  & = & \int \frac{\frac{\partial p \left( x ; \theta \right)}{\partial
  \theta}}{p \left( x ; \theta \right)} p \left( x ; \theta \right) \mathrm{d}
  x\\
  & = & \int \frac{\partial \log p \left( x ; \theta \right)}{\partial
  \theta} p \left( x ; \theta \right) \mathrm{d} x\\
  & = & E \left[ \frac{\partial \ell \left( \theta ; x \right)}{\partial
  \theta} \right]
\end{eqnarray*}
$$
**Thus the expectation of the score is zero.**

## Elastic Weight Consolidation (EWC)

Continual Learning 笔记: EWC / Online EWC / IS / MAS - Renovamen的文章 - 知乎 https://zhuanlan.zhihu.com/p/205073566

终身持续学习-可塑权重巩固（Elastic Weight Consolidation） - Glimmer的文章 - 知乎 https://zhuanlan.zhihu.com/p/86365066

作者有一处错误：
$$
\sigma^2 = -\frac{1}{f^{''}(\theta_A^*)}
$$
**Kernel**
$$
precision = \frac{1}{\sigma^2}=- E \left[ \frac{\partial^2 \ell \left( \theta ; x \right)}{\partial
  \theta^2} \right]=
  V \left[ \frac{\partial \ell \left( \theta ; x \right)}{\partial \theta}
  \right]
$$
由于 我们是极大似然估计的模型$\theta$, 所以用expectation 代表真实的value

**Algorithm**
$$
\theta = arg \min L_B(\theta)-\log P(\theta|D_A)
$$


$$
P(\theta|D_A) = N(\sigma,\mu)
$$


$$
log(P(\theta|D_A))= \log \frac{1}{\sqrt{2\pi}\sigma} - \frac{(\theta-\mu)^2}{2\sigma^2}
$$

$$
f(\theta)=log(P(\theta|D_A)=f(\theta_A^*)+\frac{1}{2}(\theta-\theta_A^*)f^"(\theta_A^*)
$$




$$
\mu =\theta_A^* \\
\sigma = - \frac{1}{f^"(\theta_A^*)}
$$




$$
\theta = arg \min L_B(\theta)-f(\theta_A^*)-\frac{1}{2}(\theta-\theta_A^*)f^"(\theta_A^*)
$$


$$
\theta = 
=arg \min L_B(\theta)-f(\theta_A^*)-\frac{1}{2}(\theta-\theta_A^*) E \left[ \frac{\partial^2 \ell \left( \theta ; x \right)}{\partial
  \theta^2} \right]
$$



$$
\theta  =  arg \min L_B(\theta)-f(\theta_A^*)+\frac{1}{2}(\theta-\theta_A^*)  V \left[ \frac{\partial \ell \left( \theta ; x \right)}{\partial \theta}
  \right]
$$

$$
\mathcal{L}(\theta) = \mathcal{L}_B(\theta)+\sum_i \frac{\lambda}{2}F_i(\theta_i-\theta^*_{A,i})^2
$$

