## GWGCL

This is the official implementation of the following paper:

> [Graph Contrastive Learning with Group Whitening](https://proceedings.mlr.press/v222/zhang24a/zhang24a.pdf)
> 
> Accepted by ACML 2023

We propose GWGCL, a graph contrastive learning method based on feature group whitening to achieve two key properties of contrastive learning: alignment and uniformity. GWGCL achieves the alignment by ensuring consistency between positive samples. There is no need for negative samples to participate, but rather to achieve the uniformity between samples through whitening. Because whitening has the
effect of feature divergence, it avoids the collapse of all sample representations to a single point, which is called dimensional collapse. Moreover, GWGCL can achieve better results and higher efficiency without the need for asymmetric networks, projection layers, stopping
gradients and complex loss function. 


## Dependencies

- python 3.9.15
- torch  2.3.1+cu118
- dgl    2.3.0+cu118
- ogb    1.3.6

## The main experiments

```
python main.py --dataname cora --epochs 100 --dfr 0.1 --der 0.6 --lr1 5e-4 --lr2 1e-2 --wd2 1e-4 --num_groups 32 --use_norm True
python main.py --dataname citeseer --dfr 0.0 --der 0.7 --lr2 0.05 --wd2 0.005 --num_groups 32 --epoch 50 --n_layers 1
python main.py --dataname pubmed --epochs 200 --dfr 0.2 --der 0.4 --lr2 1e-2 --wd2 1e-4 --num_groups 32
python main.py --dataname comp --num_groups 32 --epoch 50 --dfr 0.0 --der 0.7 --lr2 1e-2 --wd2 5e-4 --use_norm True
python main.py --dataname photo --num_groups 16 --epoch 50 --dfr 0.3 --der 0.7 --lr2 1e-2 --wd2 1e-3
```

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
