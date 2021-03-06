# Zero-shot knowledge distillation in deep networks

## Objective

Can we do Knowledge Distillation without (access to) training data (Zero-Shot)? 

- Data is precious and sensitive – won’t be shared 数据宝贵且敏感
- E.g. : Medical records, Biometric data, Proprietary data 医学记录 生物数据
- Federated learning – Only models are available, not data 联邦学习

## Existing methods

Knowledge Distillation (Hinton et al., 2015) enables to transfer the complex mapping functions learned by cumber- some models to relatively simpler models. 

$Teacher$ Model: Generally the Teacher models deliver excellent performance, but they can be **huge and computationally expensive**.  Hence, these models can not be deployed in limited resource environments or when real-time inference is expected.

$Student$ Model: has sub- stantially less memory footprint, requires less computation, and thereby often results in a much faster inference time than that of the much larger $Teacher$ model.

$Dark \ Knowledgde$:  It is this knowledge that helps the Teacher to generalize better and transfers to the Student via matching their soft labels (output of the soft-max layer) instead of the one-hot vector encoded labels.

## Main Idea

在没有目标数据先验知识的情况下，我们从教师模型中进行伪数据合成，教师模型作为传递集来执行蒸馏。

Our approach obtains useful prior information about the underlying data distribution in the form of $Class \  Similarities$ from the model parameters of the Teacher. Further, we successfully utilize this prior in the crafting process via modelling the output space of the $Teacher$ model as a Dirichlet distribution. We name the crafted samples $Data \ Impressions$ (DI) as these are the impressions of the training data as understood by the Teacher model.

### Knowledge Distillation

$$
L = \sum_{(x,y)\in D}L_{KD}(S(x,\theta_S,\tau),T(x,\theta_T,\tau))+\lambda L_{CE}(\widehat{y}_S,y)
$$

$L_{CE}$ is the cross-entropy loss computed on the labels $\widehat{y}_S$ predicted by the $Student$ and their corresponding ground truth labels $y$.

$L_{KD}$ is the distillation loss (e.g. cross-entropy
or mean square error) comparing the soft labels (softmax
outputs) predicted by the $Student$ against the soft labels
predicted by the $Teacher$. $T(x, θ_T)$ represents the softmax
output of the $Teacher$ and $S(x, θ_S)$ denotes the softmax output of the $Student$. Here, $\tau$ represents the softmax temperature.

### Modelling the Data in Softmax Space

Let $s \sim p(s)$, be the **random vector that represents the neural softmax output ** of the  $Teacher$, $T(x,\theta_T)$. We model $p(s^k)$ belong ing to each class $k$, using a **Dirichlet distribution** which is a distribution over vectors whose components are in [0,1] range and their sum is 1. Thus, the distribution to represent the softmax outputs $s^k$ of class $k$ would be modelled as, $Dir(K, \alpha ^ k)$, where $k \in {1 . . .K}$ is the class index, $K$ is the dimension of the output probability vector (number of categories in the recognition problem) and $\alpha ^ k$ is the concentration parameter of the distribution modelling class $k$. The concentration parameter $\alpha ^k$ is a $K$ dimensional positive real vector, i.e, $\alpha^k = [\alpha ^k_1, \alpha ^k_2, . . . , \alpha ^k_K]$, and $\alpha_i^k > 0$ , $\forall i$.

**Concentration Parameter ($\alpha$):**

由于狄利克雷分布的样本空间被解释为离散概率分布（在标签上），直观地说，**Concentration Parameter ($\alpha$)可以被认为是确定狄利克雷分布样本的概率质量可能“集中”的程度。**

获取Concentration Parameter的先验信息并非易事。所有组件的参数不能相同，因为这会导致所有概率集的可能性相等，这是不现实的情况。例如，在CIFAR-10数据集的情况下，dog类和plane类具有相同置信度的softmax输出是没有意义的（因为它们在视觉上是不同的）。同样，相同的$\alpha_i$值表示缺少任何先验信息来支持采样的softmax向量的一个分量而不是另一个分量。因此，应指定Concentration Parameter，以反映softmax矢量中各成分之间的相似性。由于这些成分表示识别问题中的潜在类别，因此$\alpha$应该反映它们之间的视觉相似性。	

 **Class Similarity Matrix**

The final layer of a typical recognition model will be a fully connected layer with a softmax non-linearity.

Each neuron in this layer corresponds to a class (k) and its activation is treated as the probability predicted by the model for that class.

The weights connecting the previous layer to this neuron ($w_k$) can be considered as the template of the class k learned by the $Teacher$ network.

Reason: This is because the predicted class probability is proportional to the alignment of the pre-final layer’s output with the template ($w_k$). The predicted probability peaks when the pre-final layer’s output is a positive scaled version of this template (wk). 
$$
C(i,j)=\frac{w_i^Tw_j}{||w_i||||w_j||}
$$
Since the elements of the concentration parameter have to be positive real numbers, we further perform a min-max normalization over each row of the class similarity matrix.

### Crafting Data Impression via Dirichlet Sampling

$Y^k = [y_1^k,y_2^k,...,y_N^k] \in R^{k \times N}$, be the $N$ softmax vectors corresponding to class $k$, sampled from $Dir(K,\alpha^k)$ distribution.
$$
\alpha^k = [\alpha ^k_1, \alpha ^k_2, . . . , \alpha ^k_K]
$$
Each row $c_k$ can be treated as the concentration parameter ($\alpha$) of the Dirichlet distribution (Dir), which models the distribution of output probability vectors belonging to class $k$.
$$
\alpha^k = c_k
$$

$$
C(i,j)=\frac{w_i^Tw_j}{||w_i||||w_j||}
$$

Generate $\bar{x}_i^k$ as a random noisy image

update it over multiple iterations till the cross-entropy loss between the sampled softmax vector ($y_i^k$) and the softmax output predicted by the Teacher is minimized.
$$
\bar{x}_i^k = \mathop{argmin}\limits_{x} \ L_{CE}(y_i^k,T(x,\theta_T,\tau))
$$
**Scaling Factor($\beta$)**

The probability density function of the Dirichlet distribution for $K$ random variables is a $K − 1$ dimensional probability simplex that exists on a $K$ dimensional space.

When $\alpha_i < 1,\forall i \in [1,K]$:

the density congregates at the edges of the simplex

When $\alpha_i > 1,\forall i \in [1,K]$:

the density becomes more concentrated on the center of the simplex

> Thus, we define a scaling factor (β) which can control the range of the individual elements of the concentration parameter, which in turn decides regions in the simplex from which sampling is performed.

The actual sampling of the probability vectors happen from:
$$
p(s)=Dir(K,\beta \times \alpha)
$$
$\beta$ controls the $l_1$-norm of the final concentration parameter
which, in turn, is inversely related to the variance of the
distribution.

### Zero-Shot Knowledge Distillation

我们忽略了一般蒸馏目标中的交叉熵损失，因为性能只有很小的改善或没有改善，并且它减少了超参数$\lambda$的负担
$$
\theta_s = \mathop{argmin}\limits_{\theta_s}\sum_{\bar{x}\in\bar{X}}L_{KD}(T(\bar{x},\theta_T,\tau),S(\bar{x},\theta_S,\tau))
$$
Generate a diverse set of pseudo training examples that can provide with enough information to train the Student model via Dirichlet sampling.

## Experiments

### Datasets

MNIST,Fashion MNIST,CIFAR-10

As all the experiments in these three datasets are dealing with classification problems with 10 categories each, value of the parameter $K$ in all our experiments is 10.

## Experiment Detail

Two scaling factors: $\beta_1 = 1.0 \ \beta_2=0.1$. For each dataset, half the Data Impressions are generated with $\beta_1$ and the other with $\beta_2$.

A temperature value ($\tau$) of 20 is used across all the datasets.

We augment the samples using **regular operations such as scaling, trans- lation, rotation, flipping etc**. which has proven useful in further boosting the model performance (Dao et al., 2018).

### Experiment Results

#### MNIST

Lenet-5 for $Teacher $ model; 61706 parameters

Lenet-5-Half for $Student$ model; 35820 parameters

| Model                                              | Performance | Explanation                                                  |
| -------------------------------------------------- | ----------- | ------------------------------------------------------------ |
| Teacher-CE                                         | 99.34       | The classification accuracy of the Teacher model trained using the cross-entropy (CE) loss |
| Student-CE                                         | 98.92       | The performance of the Student model trained with all the training samples and their ground truth labels using cross-entropy loss |
| Student-KD (Hinton et al., 2015) 60K original data | 99.25       | The accuracy of the Student model trained using the actual training samples through Knowledge Distillation (KD) from Teacher. |
| (Kimura et al., 2018) 200 original data            | 86.70       |                                                              |
| (Lopes et al., 2017) (uses meta data)              | 92.47       |                                                              |
| ZSKD (Ours)(24000 DIs, and no original data)       | **98.77**   | Outperform the existing few data (Kimura et al., 2018) and data-free counterparts (Lopes et al., 2017) by a great margin. It performs close to the full data (classical) Knowledge Distillation while using only 24000 DIs, i.e., 40% of the the original training set size. |

#### Fashion MNIST

Lenet-5 for $Teacher $ model; 61706 parameters

Lenet-5-Half for $Student$ model; 35820 parameters

| Model                                              | Performance |
| -------------------------------------------------- | ----------- |
| Teacher-CE                                         | 90.84       |
| Student-CE                                         | 89.43       |
| Student-KD (Hinton et al., 2015) 60K original data | 89.66       |
| (Kimura et al., 2018) 200 original data            | 72.50       |
| ZSKD (48000 DIs, and no original data)             | 79.62       |

#### CIFAR-10

AlexNet as $Teacher$ model;

考虑到数据集的复杂程度，用了更大了迁移数据集包括了40000个DI的样本，尽管依旧是低于20$\%$ 的数据集比例。

### Size of the Transfer Set

Different number of Data Impressions such as 1%, 5%, 10%, . . . , 80% of the training set size.

随着数据集变得复杂，需要生成更多的数据印象来捕获数据集中的底层模式。注意，在蒸馏过程中也观察到与实际训练样本类似的趋势。

我们观察到，所提出的输出空间的Dirichlet模型和重构的印象始终比同类模型有很大的优势。此外，在类印象的情况下，与数据印象相比，由于传输集大小增加而导致的性能增加相对较小。请注意，为了更好地理解，在进行蒸馏时显示的结果没有任何数据扩充。

## Aside

### $\beta$分布

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

## **Relatvie Source**

https://github.com/vcl-iisc/ZSKD

[Presentation](https://icml.cc/media/Slides/icml/2019/grandball(13-11-00)-13-11-30-4371-zero-shot_knowl.pdf)

[Poster](https://drive.google.com/file/d/1ZMCUPnJ3epCtLov26mVttmJT5OQB2HwK/view)