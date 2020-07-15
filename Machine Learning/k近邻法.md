# k近邻算法

Step1：
$$
T = {(x_1,y_1),(x_2,y_2),...(x_N,y_N)}
$$



$$
x_i \in \mathcal{X} \subseteq R^n 
$$

$$
y_i \in \mathcal{Y}=\{c_1,c_2,...c_k\}  i = 1,2,..N
$$


Step2：

根据给定的距离度量，在训练集T重找出与$x$最邻近的$k$个点，涵盖着$k$个点的$x$的领域记作$N_k(x)$
$$
y = arg \max_{c_j} \sum_{x_i \in N_k(x)}I(y_i = c_j),i=1,2,...N
$$

# k近邻模型

## 模型

**k近邻模型**实质上是一个空间划分模型。根据训练样本自身的特征，通过距离公式计算，将训练数据集组成空间整体划分成M个字空间（M为类别数）。利用测试集进行测试评估模型的好快，以调整k的选择或者距离方法的选择。在此，经常使用交叉验证的方法。

## 距离度量

**特征空间**中两个实例点的距离是两个实例点相似程度的反映。在此我们介绍一些我们经常用到的一些距离公式：

Assumption：
$$
设特征空间\mathcal{X}是n维实数向量空间R^n，x_i,x_j \in \mathcal{X},x_i=(x_i^{(1)},x_i^{(2)},..x_i^{(n)})^T,x_j=(x_j^{(1)},x_j^{(2)},..x_j^{(n)})^T,x_i,x_j的L_p距离定义为：
$$

$$
L_p(x_i,x_j)=(\sum_{l=1}^n|x_i^{(l)}-x_j^{(l)}|^p)^{\frac{1}{p}}
$$




其中当$p=2$，称为欧式距离(Euclidean distance)：
$$
L_2(x_i,x_j)=(\sum_{l=1}^n|x_i^{(l)}-x_j^{(l)}|^2)^{\frac{1}{2}}
$$
其中当$p=1$，称为曼哈顿距离(Manhattan distance)：
$$
L_2(x_i,x_j)=\sum_{l=1}^n|x_i^{(l)}-x_j^{(l)}|
$$
其中当$p=\infty$，称为曼哈顿距离(Manhattan distance)：
$$
L_{\infty}(x_i,x_j) = \max_l|x_i^{(l)}-x_j^{(l)}|
$$

## k值的选择









































