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






























