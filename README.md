## GWGCL

This is the official implementation of the following paper:

> [Graph Contrastive Learning with Group Whitening](https://proceedings.mlr.press/v222/zhang24a/zhang24a.pdf)
> 
> Accepted to ACML 2023

We propose GWGCL, a graph contrastive learning method based on feature group whitening to achieve two key properties of contrastive learning: alignment and uniformity. GWGCL achieves the alignment by ensuring consistency between positive samples. There is no need for negative samples to participate, but rather to achieve the uniformity between samples through whitening. Because whitening has the
effect of feature divergence, it avoids the collapse of all sample representations to a single point, which is called dimensional collapse. Moreover, GWGCL can achieve better results and higher efficiency without the need for asymmetric networks, projection layers, stopping
gradients and complex loss function. 
