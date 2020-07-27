# Gamma函数

## Gamma 函数定义

Gamma函数如下：

![[公式]](https://www.zhihu.com/equation?tex=%5CGamma%28%5Calpha%29%3D%5Cint+_%7B0%7D%5E%7B%5Cinfty%7Dt%5E%7B%5Calpha-1%7De%5E%7B-t%7Ddt%2C+%5Calpha%3E0%5C%5C)

很奇怪，但可以形象理解为用一个伽马刀，对 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 动了一刀，于是指数为 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha-1) ,动完刀需要扶着梯子 ![[公式]](https://www.zhihu.com/equation?tex=%28-t%29) 才能走下来

通过分布积分可以得到如下性质：

![[公式]](https://www.zhihu.com/equation?tex=%5CGamma%28%5Calpha%2B1%29%3D%5Cint+_%7B0%7D%5E%7B%5Cinfty%7Dt%5E%7B%5Calpha%7De%5E%7B-t%7Ddt%3D-%5Cint+_%7B0%7D%5E%7B%5Cinfty%7Dt%5E%7B%5Calpha%7Dd%28e%5E%7B-t%7D%29%3D-%5Cleft%5Bt%5E%7B%5Calpha%7De%5E%7B-t%7D%7C_%7B0%7D%5E%7B%5Cinfty%7D-%5Calpha%5Cint_%7B0%7D%5E%7B%5Cinfty%7De%5E%7B-t%7Dt%5E%7B%5Calpha-1%7Ddt%5Cright%5D%3D%5Calpha%5CGamma%28%5Calpha%29%5C%5C)
易证明有如下性质：

![[公式]](https://www.zhihu.com/equation?tex=%5CGamma%28n%2B1%29%3Dn%21%2C%5CGamma%281%29%3D1%2C%5CGamma%28%5Cfrac%7B1%7D%7B2%7D%29%3D%5Csqrt%7B%5Cpi%7D%5C%5C)



































