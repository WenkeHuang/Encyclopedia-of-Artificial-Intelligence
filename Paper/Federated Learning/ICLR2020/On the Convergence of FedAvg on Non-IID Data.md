# On the Convergence of FedAvg on Non-IID Data

## Problems

FL has **three** unique characters that distinguish it from the standard parallel optimization

- First, the training data are massively distributed over an incredibly large number of devices, and the connection between the central server and a device is slow.  

  训练数据很多并且分布在很大规模的蛇别上，中央服务器和设备之间的连接的慢的。

- Second, unlike the traditional distributed learning systems, the FL system does not have control over users’ devices. 

  其次，相比于传统的分布式学习，联邦学习系统不能控制用户设备

- Third, the training data are non-iid, that is, a device’s local data cannot be regarded as samples drawn from the overall distribution.

  训练数据也是非独立同分布的，一个用户的本地数据不能被看作是从整体分布中采样出来的

## Related Work

There have been much efforts developing convergence guarantees for FL algorithm based on the assumptions that 

1. **the data are iid** 
2. **all the devices are active.**







































