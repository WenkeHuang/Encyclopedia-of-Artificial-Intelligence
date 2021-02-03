# Different Distribution

## Bernoulli Distribution

伯努利分布(*Bernoulli distribution*)又名**两点分布**或**0-1分布**，介绍伯努利分布前首先需要引入**伯努利试验（Bernoulli trial）**。

 伯努利试验是只有两种可能结果的单次随机试验，即对于一个随机变量X而言：
$$
P_r[x=1]=p \\
P_r[x=0] = 1-p
$$

- 如果试验E是一个伯努利试验，将E独立重复地进行n次，则称这一串重复的独立试验为**n重伯努利试验**。

- 进行一次伯努利试验，成功(X=1)概率为p(0<=p<=1)，失败(X=0)概率为1-p，则称随机变量X服从伯努利分布。伯努利分布是离散型概率分布，其概率质量函数为：
  $$
  f(x)=p(x)(1-p)^{1-x}=\begin{cases}
  p \ \  if \ x=1 \\
  1-p \ \ if \ x=0 \\
  0 \ \ otherwise
  \end{cases}
  $$
  

## Binomial Distribution

**二项分布**是指在只有两个结果的n次独立的伯努利试验中，所期望的结果出现**次数的概率**。在单次试验中，结果A出现的概率为p，结果B出现的概率为q，p+q=1。那么在n=10，即10次试验中，结果A出现0次、1次、……、10次的概率各是多少呢？这样的概率分布呈现出什么特征呢？这就是二项分布所研究的内容。
$$
b(x,n,p)=C_n^x p^x q^{n-x}
$$
其中b表示二项分布的概率，n表示试验次数，x表示出现某个结果的次数。是组合，表示在n次试验中出现x次结果的可能的次数。如10次试验，出现0次正面的次数有1次，出现1次正面的次数有10次，……，出现5次正面的次数有252次，等等。其计算也有一个通式：
$$
C_n^x = \frac{n \times (n-1) \times ... \times (n-x+1)}{x \times (x-1) \times ...\times 1} = \frac{n!}{(n-x)!x!}
$$

## Multinomial Distribution

多项式分布(*Multinomial Distribution*)是二项式分布的推广。二项式做n次伯努利实验，规定了每次试验的结果只有两个，如果现在还是做n次试验，只不过每次试验的结果可以有多m个，且m个结果发生的概率互斥且和为1，则发生其中一个结果X次的概率就是多项式分布。

**扔骰子是典型的多项式分布**。扔骰子，不同于扔硬币，骰子有6个面对应6个不同的点数，这样单次每个点数朝上的概率都是1/6（对应p1~p6，它们的值不一定都是1/6，只要和为1且互斥即可，比如一个形状不规则的骰子）,重复扔n次，如果问有k次都是点数6朝上的概率就是
$$
P(X=k) = C_n^k p_6^k(1-p_6)^{n-k},k=0,1,2,...,n
$$
 多项式分布一般的概率质量函数为：
$$
P\{ X_1=k_1,X_2=k_2,..,X_n=k_n\}=\frac{n!}{k_1! k_2!...k_n!}\prod_{i=1}^np_i^{k_i} where \sum{i=0}^n k_i=n
$$

## $\beta$分布

- 通俗的讲，**先验概率**就是事情尚未发生前，我们对该事发生概率的估计。利用过去历史资料计算得到的先验概率，称为**客观先验概率**； 当历史资料无从取得或资料不完全时，凭人们的主观经验来判断而得到的先验概率，称为**主观先验概率**。例如抛一枚硬币头向上的概率为0.5，这就是主观先验概率。

- **后验概率**是指通过调查或其它方式获取新的附加信息，利用贝叶斯公式对先验概率进行修正，而后得到的概率。



$Beta(\alpha,\beta)$

Parameters:

$\alpha > 0 \ \beta>0 \ x \in [0,1]$

The probability density function (pdf) of the beta distribution, for $0 \leq x \leq 1$, and shape parameters $\alpha \ \beta>  0$, is a power function of the variable *x* and of its reflection (1 − *x*) as follows:
$$
f(x;\alpha,\beta) = constant x^{\alpha-1}(1-x)^{\beta-1} 
$$

$$
=\frac{x^{\alpha-1}(1-x)^{\beta-1}}{\int_0^1u^{\alpha-1}(1-u)^{\beta-1}du}
$$

$$
= \frac{\Gamma (\alpha + \beta)}{\Gamma(\alpha)\Gamma (\beta)} x^{\alpha-1}(1-x)^{\beta-1}
$$

$$
=\frac{1}{B(\alpha,\beta)}x^{\alpha-1}(1-x)^{\beta-1}
$$

$$
E[X]=\frac{\alpha}{\alpha+\beta}
$$

Where $\Gamma$ is the Gamma function：伽玛函数（Gamma Function）作为阶乘的延拓，是定义在复数范围内的亚纯函数。

（1）在实数域上伽玛函数定义为：
$$
\Gamma(x) = \int_0^{+\infty}t^{x-1}e^{-t}dt(x>0)
$$
对于任何正整数$n$有：
$$
\Gamma(n) = (n-1)!
$$
（2）在复数域上伽玛函数定义为：
$$
\Gamma(x) = \int_0^{+\infty}t^{x-1}e^{-t}dt
$$


### 狄利克雷分布

狄利克雷分布是一种“分布的分布” (a distribution on probability distribution) ，由两个参数$\alpha$，$G_0$确定，即$G\sim DP(\alpha，G_0)$， $\alpha$是分布参数(concentration or scaling parameter)，**其值越大，分布越接近于均匀分布**，其值越小，分布越concentrated。$G_0$是基分布(base distribution)。

可以把DP想象成黑箱，输入分布 $G_0$是，输出分布$G$，而 $\alpha$ 控制输出的样子

**问题背景**

我们有一组来源于混合高斯分布的数据集，希望对其进行聚类，然而我们并不知道这组数据是由几组高斯分布生成的。

**问题特点**

1. 聚类数量未知
2. 非参数化，即不确定参数，如果需要，参数数量可以变化
3. 聚类数量服从概率分布

**可行方法**

针对高斯混合模型(Gaussian Mixture Models)做最大期望运算(Expectation Maximization, EM)，分析结果，继续迭代计算。也可以做层次聚类(Hierarchical Clustering)，比如层次凝聚法(Hierarchical Agglomerative Clustering, HAC)，再进行人为剪枝。

然而，最希望的还是用一种**以统计学为主，尽量避免主管因素**（比如人为规定类别数量，人为进行剪枝）的方法来对数据进行聚类。



https://www.zhihu.com/question/26751755

#  Reference:

https://leimao.github.io/blog/Introduction-to-Dirichlet-Distribution/

