## GWGCL

This is the official implementation of the following paper:

> [Graph Contrastive Learning with Group Whitening](https://proceedings.mlr.press/v222/zhang24a/zhang24a.pdf)
> 
> Accepted by ACML 2023

We propose GWGCL, a graph contrastive learning method based on feature group whitening to achieve two key properties of contrastive learning: alignment and uniformity. GWGCL achieves the alignment by ensuring consistency between positive samples. There is no need for negative samples to participate, but rather to achieve the uniformity between samples through whitening. Because whitening has the
effect of feature divergence, it avoids the collapse of all sample representations to a single point, which is called dimensional collapse. Moreover, GWGCL can achieve better results and higher efficiency without the need for asymmetric networks, projection layers, stopping
gradients and complex loss function. 

## Citation
If you find our repository useful for your research, please consider citing our paper:
```
@inproceedings{zhang2024graph,
  title={Graph Contrastive Learning with Group Whitening},
  author={Zhang, Chunhui and Miao, Rui},
  booktitle={Asian Conference on Machine Learning},
  pages={1622--1637},
  year={2023},
  organization={PMLR}
}
```
