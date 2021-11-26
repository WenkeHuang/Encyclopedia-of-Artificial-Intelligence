# A simple framework for contrastive learning of visual representations

## Idea

A simple idea: maximizing the agreement of representations under data transformation, using a contrastive loss in the latent/feature space. 

1.  先sample一些图片组为batch；
2.  对batch里的image做两种不同的data augmentation；
3. 希望同一张图像、不同augmentation的结果相近，并互斥其他结果。

### Loss function

Let $sim(u,v)=u^T/||u||||v||$
$$
\mathcal{l}_{i,j}=-log\frac{exp(sim(z_i,z_j))}{\sum+{k=1}^{2N}\mathbb{I}_{[k\neq i]}exp(sim(z_i,z_k)/\tau)}
$$


## Conclusion

- SimCLR is a simple yet effective self-supervised learning framework, advancing state-of-the-art by a large margin. 
- The superior performance of SimCLR is not due to any single design choice, but a combination of design choices. 
- Our studies reveal several important factors that enable effective representation learning, which could help future research.

