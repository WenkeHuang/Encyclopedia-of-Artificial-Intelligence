# 循环序列模型（Recurrent Neural Networks）

## 为什么选择序列模型

他们都只能单独的取处理一个个的输入，前一个输入和后一个输入是完全没有关系的。但是，某些任务需要能够更好的处理**序列**的信息，即前面的输入和后面的输入是有关系的。

> 比如，当我们在理解一句话意思时，孤立的理解这句话的每个词是不够的，我们需要处理这些词连接起来的整个序列；当我们处理视频的时候，我们也不能只单独的去分析每一帧，而要分析这些帧连接起来的整个序列。

## RNN结构

首先看一个简单的循环神经网络如，它由输入层、一个隐藏层和一个输出层组成：

<img src="../img/DL/RNNstructure.jpg" alt="RNNstructure" style="zoom:80%;" />

我们现在这样来理解，如果把上面有W的那个带箭头的圈去掉，它就变成了最普通的**全连接神经网络**。x是一个向量，它表示**输入层**的值（这里面没有画出来表示神经元节点的圆圈）；s是一个向量，它表示**隐藏层**的值（这里隐藏层面画了一个节点，你也可以想象这一层其实是多个节点，节点数与向量s的维度相同）；

U是输入层到隐藏层的**权重矩阵**，o也是一个向量，它表示**输出层**的值；V是隐藏层到输出层的**权重矩阵**。

**循环神经网络**的**隐藏层**的值s不仅仅取决于当前这次的输入x，还取决于上一次**隐藏层**的值s。**权重矩阵** W就是**隐藏层**上一次的值作为这一次的输入的权重。

<img src="../img/DL/RNNdetailstructure.jpg" alt="RNNdetailstructure" style="zoom:80%;" />

如果我们把上面的图展开，**循环神经网络**也可以画成下面这个样子：

<img src="../img/DL/RNNstructuretime.jpg" alt="RNNstructuretime" style="zoom:80%;" />

现在看上去就比较清楚了，这个网络在t时刻接收到输入 ![[公式]](https://www.zhihu.com/equation?tex=x_%7Bt%7D) 之后，隐藏层的值是 ![[公式]](https://www.zhihu.com/equation?tex=s_%7Bt%7D) ，输出值是 ![[公式]](https://www.zhihu.com/equation?tex=o_%7Bt%7D) 。关键一点是， ![[公式]](https://www.zhihu.com/equation?tex=s_%7Bt%7D) 的值不仅仅取决于 ![[公式]](https://www.zhihu.com/equation?tex=x_%7Bt%7D) ，还取决于 ![[公式]](https://www.zhihu.com/equation?tex=s_%7Bt-1%7D) 。我们可以用下面的公式来表示**循环神经网络**的计算方法：

用公式表示如下
$$
O_t = g(V\cdot S_t)\\
S_t = f(U \cdot X_t + W \cdot S_{t-1})
$$

## RNN 梯度消失

无论是梯度消失还是梯度爆炸，都是**源于网络结构太深**，造成网络权重不稳定，从本质上来讲是**因为梯度反向传播中的连乘效应。**

我们给定一个三个时间的RNN单元，如下：

<img src="../img/DL/RNNgradient.jpg" alt="RNNgradient" style="zoom:80%;" />

我们假设最左端的输入 ![[公式]](https://www.zhihu.com/equation?tex=S_0) 为给定值， 且神经元中没有激活函数（便于分析）， 则前向过程如下：


![[公式]](https://www.zhihu.com/equation?tex=S_1+%3D+W_xX_1+%2B+W_sS_0+%2B+b_1+%5Cqquad+%5Cqquad+%5Cqquad+O_1+%3D+W_oS_1+%2B+b_2+%5C%5C+S_2+%3D+W_xX_2+%2B+W_sS_1+%2B+b_1+%5Cqquad+%5Cqquad+%5Cqquad+O_2+%3D+W_oS_2+%2B+b_2+%5C%5C+S_3+%3D+W_xX_3+%2B+W_sS_2+%2B+b_1+%5Cqquad+%5Cqquad+%5Cqquad+O_3+%3D+W_oS_3+%2B+b_2+%5C%5C)

在 ![[公式]](https://www.zhihu.com/equation?tex=t%3D3) 时刻， 损失函数为 ![[公式]](https://www.zhihu.com/equation?tex=L_3+%3D+%5Cfrac%7B1%7D%7B2%7D%28Y_3+-+O_3%29%5E2) ，那么如果我们要训练RNN时， 实际上就是是对 ![[公式]](https://www.zhihu.com/equation?tex=W_x%2C+W_s%2C+W_o%2Cb_1%2Cb_2) 求偏导， 并不断调整它们以使得 ![[公式]](https://www.zhihu.com/equation?tex=L_3) 尽可能达到最小（参见反向传播算法与梯度下降算法)。

那么我们得到以下公式：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cdelta+L_3%7D%7B%5Cdelta+W_0%7D+%3D+%5Cfrac%7B%5Cdelta+L_3%7D%7B%5Cdelta+O_3%7D+%5Cfrac%7B%5Cdelta+O_3%7D%7B%5Cdelta+W_0%7D+%5C%5C+%5Cfrac%7B%5Cdelta+L_3%7D%7B%5Cdelta+W_x%7D+%3D+%5Cfrac%7B%5Cdelta+L_3%7D%7B%5Cdelta+O_3%7D+%5Cfrac%7B%5Cdelta+O_3%7D%7B%5Cdelta+S_3%7D+%5Cfrac%7B%5Cdelta+S_3%7D%7B%5Cdelta+W_x%7D+%2B+%5Cfrac%7B%5Cdelta+L_3%7D%7B%5Cdelta+O_3%7D+%5Cfrac%7B%5Cdelta+O_3%7D%7B%5Cdelta+S_3%7D+%5Cfrac%7B%5Cdelta+S_3%7D%7B%5Cdelta+S_2%7D+%5Cfrac%7B%5Cdelta+S_2%7D%7B%5Cdelta+W_x%7D+%2B+%5Cfrac%7B%5Cdelta+L_3%7D%7B%5Cdelta+O_3%7D+%5Cfrac%7B%5Cdelta+O_3%7D%7B%5Cdelta+S_3%7D+%5Cfrac%7B%5Cdelta+S_3%7D%7B%5Cdelta+S_2%7D+%5Cfrac%7B%5Cdelta+S_2%7D%7B%5Cdelta+S_1%7D%5Cfrac%7B%5Cdelta+S_1%7D%7B%5Cdelta+W_x%7D+%5C%5C+%5Cfrac%7B%5Cdelta+L_3%7D%7B%5Cdelta+W_s%7D+%3D+%5Cfrac%7B%5Cdelta+L_3%7D%7B%5Cdelta+O_3%7D+%5Cfrac%7B%5Cdelta+O_3%7D%7B%5Cdelta+S_3%7D+%5Cfrac%7B%5Cdelta+S_3%7D%7B%5Cdelta+W_s%7D+%2B+%5Cfrac%7B%5Cdelta+L_3%7D%7B%5Cdelta+O_3%7D+%5Cfrac%7B%5Cdelta+O_3%7D%7B%5Cdelta+S_3%7D+%5Cfrac%7B%5Cdelta+S_3%7D%7B%5Cdelta+S_2%7D+%5Cfrac%7B%5Cdelta+S_2%7D%7B%5Cdelta+W_s%7D+%2B+%5Cfrac%7B%5Cdelta+L_3%7D%7B%5Cdelta+O_3%7D+%5Cfrac%7B%5Cdelta+O_3%7D%7B%5Cdelta+S_3%7D+%5Cfrac%7B%5Cdelta+S_3%7D%7B%5Cdelta+S_2%7D+%5Cfrac%7B%5Cdelta+S_2%7D%7B%5Cdelta+S_1%7D%5Cfrac%7B%5Cdelta+S_1%7D%7B%5Cdelta+W_s%7D+%5C%5C)

将上述偏导公式与第三节中的公式比较，我们发现， 随着神经网络层数的加深对 ![[公式]](https://www.zhihu.com/equation?tex=W_0) 而言并没有什么影响， 而对 ![[公式]](https://www.zhihu.com/equation?tex=W_x%2C+W_s) 会随着时间序列的拉长而产生梯度消失和梯度爆炸问题。

根据上述分析整理一下公式可得， 对于任意时刻t对 ![[公式]](https://www.zhihu.com/equation?tex=W_x%2C+W_s) 求偏导的公式为：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cdelta+L_t%7D%7B%5Cdelta+W_x+%7D+%3D+%5Csum_%7Bk%3D0%7D%5Et+%5Cfrac%7B%5Cdelta+L_t%7D%7B%5Cdelta+O_t%7D+%5Cfrac%7B%5Cdelta+O_t%7D%7B%5Cdelta+S_t%7D%28+%5Cprod_%7Bj%3Dk%2B1%7D%5Et+%5Cfrac%7B%5Cdelta+S_j%7D%7B%5Cdelta+S_%7Bj-1%7D%7D+%29+%5Cfrac%7B+%5Cdelta+S_k+%7D%7B%5Cdelta+W_x%7D+%5C%5C+%5Cfrac%7B%5Cdelta+L_t%7D%7B%5Cdelta+W_s+%7D+%3D+%5Csum_%7Bk%3D0%7D%5Et+%5Cfrac%7B%5Cdelta+L_t%7D%7B%5Cdelta+O_t%7D+%5Cfrac%7B%5Cdelta+O_t%7D%7B%5Cdelta+S_t%7D%28+%5Cprod_%7Bj%3Dk%2B1%7D%5Et+%5Cfrac%7B%5Cdelta+S_j%7D%7B%5Cdelta+S_%7Bj-1%7D%7D+%29+%5Cfrac%7B+%5Cdelta+S_k+%7D%7B%5Cdelta+W_s%7D)

我们发现， 导致梯度消失和爆炸的就在于 ![[公式]](https://www.zhihu.com/equation?tex=%5Cprod_%7Bj%3Dk%2B1%7D%5Et+%5Cfrac%7B%5Cdelta+S_j%7D%7B%5Cdelta+S_%7Bj-1%7D%7D) ， 而加上激活函数后的S的表达式为：

![[公式]](https://www.zhihu.com/equation?tex=S_j+%3D+tanh%28W_xX_j+%2B+W_sS_%7Bj-1%7D+%2B+b_1%29+)

那么则有：

![[公式]](https://www.zhihu.com/equation?tex=%5Cprod_%7Bj%3Dk%2B1%7D%5Et+%5Cfrac%7B%5Cdelta+S_j%7D%7B%5Cdelta+S_%7Bj-1%7D%7D+%3D+%5Cprod_%7Bj%3Dk%2B1%7D%5Et+tanh%27+W_s)

而在这个公式中， tanh的导数总是小于1 的， 如果 ![[公式]](https://www.zhihu.com/equation?tex=W_s) 也是一个大于0小于1的值， 那么随着t的增大， 上述公式的值越来越趋近于0， 这就导致了梯度消失问题。 那么如果 ![[公式]](https://www.zhihu.com/equation?tex=W_s) 很大， 上述公式会越来越趋向于无穷， 这就产生了梯度爆炸。

## LSTM

RNN产生梯度消失与梯度爆炸的原因就在于 ![[公式]](https://www.zhihu.com/equation?tex=%5Cprod_%7Bj%3Dk%2B1%7D%5Et+%5Cfrac%7B%5Cdelta+S_j%7D%7B%5Cdelta+S_%7Bj-1%7D%7D+) ， 如果我们能够将这一坨东西去掉， 我们的不就解决掉梯度问题了吗。 LSTM通过门机制来解决了这个问题。

我们先从LSTM的三个门公式出发：

- 遗忘门： ![[公式]](https://www.zhihu.com/equation?tex=f_t+%3D+%5Csigma%28+W_f+%5Ccdot+%5Bh_%7Bt-1%7D%2C+x_t%5D+%2B+b_f%29)
- 输入门： ![[公式]](https://www.zhihu.com/equation?tex=i_t+%3D+%5Csigma%28W_i+%5Ccdot+%5Bh_%7Bt-1%7D%2C+x_t%5D+%2B+b_i%29)
- 输出门： ![[公式]](https://www.zhihu.com/equation?tex=o_t+%3D+%5Csigma%28W_o+%5Ccdot+%5Bh_%7Bt-1%7D%2C+x_t+%5D+%2B+b_0+%29)
- 当前单元状态 ![[公式]](https://www.zhihu.com/equation?tex=c_t) : ![[公式]](https://www.zhihu.com/equation?tex=c_t+%3D+f_t+%5Ccirc+c_%7Bt-1%7D+%2B+i_t+%5Ccirc+tanh%28W_c+%5Ccdot+%5Bh_%7Bt-1%7D%2C+x_t%5D+%2B+b_c+%29)
- 当前时刻的隐层输出： ![[公式]](https://www.zhihu.com/equation?tex=h_t+%3D+o_t+%5Ccirc+tanh%28c_t%29)

我们注意到， 首先三个门的激活函数是sigmoid， 这也就意味着这三个门的输出要么接近于0 ， 要么接近于1。这就使得 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cdelta+c_t%7D%7B%5Cdelta+c_%7Bt-1%7D%7D+%3D+f_t%EF%BC%8C+%5Cfrac%7B%5Cdelta+h_t%7D%7B%5Cdelta+h_%7Bt-1%7D%7D+%3D+o_t) 是非0即1的，当门为1时， 梯度能够很好的在LSTM中传递，很大程度上减轻了梯度消失发生的概率， 当门为0时，说明上一时刻的信息对当前时刻没有影响， 我们也就没有必要传递梯度回去来更新参数了。所以， 这就是为什么通过门机制就能够解决梯度的原因： 使得单元间的传递 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cdelta+S_j%7D%7B%5Cdelta+S_%7Bj-1%7D%7D) 为0 或 1。

## GRU

GRU是另一种十分主流的RNN衍生物。 RNN和LSTM都是在设计网络结构用于缓解梯度消失问题，只不过是网络结构有所不同。GRU在数学上的形式化表示如下：
$$
z_t :=sigm(W_{xz}x_t+W_{hz}h_{t-1}),\\
r_t: =sigm(W_{xr}x_t+W_{hr}h_{t-1}),\\
\overline{h}_t :=tanh(W_{xh}x_t+r_t\odot(W_{hh}h_{t-1})),\\
h_t:=(1-z_t)\odot \overline{h}_t+z_t\odot h_{t-1}
$$

- **重置门** ![[公式]](https://www.zhihu.com/equation?tex=r_t) ： ![[公式]](https://www.zhihu.com/equation?tex=r_t) 用于控制前一时刻隐层单元 ![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol+h_%7Bt-1%7D) 对当前词 ![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol+x_t) 的影响。如果 ![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol+h_%7Bt-1%7D) 对 ![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol+x_t) 不重要， 即从当前词 ![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol+x_t) 开始表述了新的意思，与上文无关。那么开关 ![[公式]](https://www.zhihu.com/equation?tex=r_t) 可以打开，使得 ![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol+h_%7Bt-1%7D) 对 ![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol+x_t) 不产生影响。
- **更新门** ![[公式]](https://www.zhihu.com/equation?tex=z_t) ： ![[公式]](https://www.zhihu.com/equation?tex=z_t) 用于决定是否忽略当前词 ![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol+x_t) 。类似于LSTM中的输入门 ![[公式]](https://www.zhihu.com/equation?tex=i_t) ， ![[公式]](https://www.zhihu.com/equation?tex=z_t) 可以判断当前词 ![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol+x_t) 对整体意思的表达是否重要。当 ![[公式]](https://www.zhihu.com/equation?tex=z_t) 开关接通下面的支路时，我们将忽略当前词 ![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol+x_t) ，同时构成了从 ![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol+h_%7Bt-1%7D) 到 ![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol+h_t) 的短路连接，这使得梯度得已有效地反向传播。和LSTM相同，这种短路机制有效地缓解了梯度消失现象，这个机制于highway networks十分相似。

# 自然语言处理与词嵌入（Natural Language Processing and Word Embeddings） 

## 词汇表征 Word Representation

一般地，第一步我们会将每一个单词转换为一个onehot向量，假如我们使用的词典有10000个单词，那么每个单词转换为onehot向量后都是一个10000*1的向量，这个向量中只有单词所在的位置为1其余位置都为0；第二步便可以将这三个单词进行比较相似度了（采用余弦相似度或者欧式距离等）。如不出意外，我们发现这三个单词两两的相似度是相同的，那这是否意味着这3个句子都是不相关或者相关度是一样的呢？显然不是的。

在将这三个句子进行聚类时，我们首先将这个句子中的每个单词进行了onehot向量化，但是这个onehot向量把每个词都孤立起来，这样使得算法对相关词的泛化能力不强。例如apple和orange都属于水果，他们的形状都是圆的，都是可以可吃的等等具有多个相似性的特征，而iPhone属于手机，形状一般是方形，但是向量化onehot后并不能将这些特征表示出来，它只是显示出他们在词典中的位置。显然，如果我们能换一种表示，将这些单词的特征都表示出来，对我们的聚类无疑是非常有帮助的。

词嵌入（word embedding）便是这样将一个单词的表示特征化的技术。词嵌入相当于将一个单词嵌入到一个多维的特征矩阵中，在每个维度的的值代表着这个单词在这个维度的特征。下面通过另一个例子来更具体的看看词嵌入是怎么表达出词的相关性的。

![wordrepresentation](../img/DL/wordrepresentation.png)

















































