https://wiseodd.github.io/techblog/2018/03/11/fisher-information/

Fisher information的直观定义就是观测数据蕴含的信息量

如果从定义出发，Fisher information代表给定数据下，似然函数对估计参数的敏感度。敏感度越大，越有利于估计。类似的理解也可以从deviance的角度来看
$$
I(\theta)=E_\theta\left[-\frac{\partial^2 }{\partial \theta^2}\ln P(\theta;X)\right].
$$
You use the information when you want to conduct inference by maximizing the log likelihood. That log-likelihood is a function of $\theta$ that is random because it depends on $X$.



**Equating the score**
$$
\frac{\partial\ell \left( \theta ; x \right)}{\partial \theta} = \frac{\partial\log p \left( x ; \theta \right)}{\partial \theta}
$$
Now, we know that on average, the score is zero (see proof of that point at the end of this answer). Thus
$$
\begin{eqnarray*}
  E \left[ \frac{\partial \ell \left( \theta ; x \right)}{\partial \theta}
  \right] & = & 0\\
  \int \frac{\partial \ell \left( \theta ; x \right)}{\partial \theta} p
  \left( x ; \theta \right) d x & = & 0
\end{eqnarray*}
$$
Take **derivatives** at both sides (we can interchange integral and derivative here but I am not going to give rigorous conditions here)
$$
\begin{eqnarray*}
  \frac{\partial}{\partial \theta} \int \frac{\partial \ell \left( \theta ; x
  \right)}{\partial \theta} p \left( x ; \theta \right) d x & = & 0\\
  \int \frac{\partial^2 \ell \left( \theta ; x \right)}{\partial \theta^2} p
  \left( x ; \theta \right) d x + \int \frac{\partial \ell \left( \theta
  ; x \right)}{\partial \theta} \frac{\partial p \left( x ; \theta
  \right)}{\partial \theta} d x & = & 0
\end{eqnarray*}
$$
The second term on the left-hand side is
$$
\begin{eqnarray*}
  \int \frac{\partial \ell \left( \theta ; x \right)}{\partial \theta}
  \frac{\partial p \left( x ; \theta \right)}{\partial \theta} d x & = &
  \int \frac{\partial \log p \left( x ; \theta \right)}{\partial \theta}
  \frac{\partial p \left( x ; \theta \right)}{\partial \theta} d x\\
  & = & \int \frac{\partial \log p \left( x ; \theta \right)}{\partial
  \theta} \frac{\frac{\partial p \left( x ; \theta \right)}{\partial
  \theta}}{p \left( x ; \theta \right)} p \left( x ; \theta \right) d x\\
  & = & \int \left( \frac{\partial \log p \left( x ; \theta \right)}{\partial
  \theta} \right)^2 p \left( x ; \theta \right) d x\\
  & = & V \left[ \frac{\partial \ell \left( \theta ; x \right)}{\partial
  \theta} \right]
\end{eqnarray*}
$$
于是得到了**Fisher Information的第一条数学意义：就是用来估计MLE的方程的方差**。它的直观表述就是，随着收集的数据越来越多，这个方差由于是一个Independent sum的形式，也就变的越来越大，也就象征着得到的信息越来越多。

here the second follows from dividing and multiplying by $p(x;\theta)$ 

The third line follows from applying the chain rule to derivative of log. The final line follows from the expectation of the score being zero, that is the variance is equal to the expectation of the square and no need to subtract the square of the expectation.)
$$
\frac{\frac{\partial p \left( x ; \theta \right)}{\partial
  \theta}}{p \left( x ; \theta \right)} = \frac{\partial \log p \left( x ; \theta \right)}{\partial
  \theta}
$$
From which you can see

而且，如果log likelihood二阶可导，在一般情况下（under specific regularity conditions）可以很容易地证明:
$$
\begin{eqnarray*}
  V \left[ \frac{\partial \ell \left( \theta ; x \right)}{\partial \theta}
  \right] & = & - \int \frac{\partial^2 \ell \left( \theta ; x
  \right)}{\partial \theta^2} p \left( x ; \theta \right) dx\\
  & = & - E \left[ \frac{\partial^2 \ell \left( \theta ; x \right)}{\partial
  \theta^2} \right]
\end{eqnarray*}
$$
于是得到了**Fisher Information的第二条数学意义：log likelihood在参数真实值处的负二阶导数的期望**。



To answer an additional question by the OP, I will show what the expectation of the score is zero. Since $p \left( x, \theta \right)$ is a density:
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
Looking on the left hand side
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







