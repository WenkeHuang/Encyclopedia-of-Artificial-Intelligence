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











































