# Momentum Contrast for Unsupervised Visual Representation Learning

既然对比是在正负例之间进行的，那负例越多，这个任务就越难，于是一个优化方向就是增加负例。

纯粹的增大batch size是不行的，总会受到GPU内存限制。一个可行的办法就是增加memory bank，把之前编码好的样本存储起来，计算loss的时候一起作为负例：

## Idea

对于每个batch x：

随机增强出 $x^q$ , $x^k$两种view

分别用$f_q$, $f_k$ 对输入进行编码得到归一化的$ q$ 和 $k$，并去掉 $k $的梯度更新

将 $q$ 和 $k $一一对应相乘得到正例的cosine（Nx1），再将 $q$ 和队列中存储的$K$个负样本相乘（$NxK$），拼接起来的到 $Nx(1+K) $大小的矩阵，这时第一个元素就是正例，直接计算交叉熵损失，更新$f_q$的参数

动量更新$f_k$  的参数：

将 $k$ 加入队列，把队首的旧编码出队，负例最多时有65536个


