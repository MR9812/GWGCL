#group whiten
#python main.py --dataname cora --epochs 100 --dfr 0.1 --der 0.6 --lr1 5e-4 --lr2 1e-2 --wd2 1e-4 --num_groups 32 --use_norm True --gpu 1
#84.48+0.81
#python main.py --dataname citeseer --epoch 50 --n_layers 1 --dfr 0.4 --der 0.6 --lr2 5e-2 --wd2 1e-2 --num_groups 16 --hid_dim 1024 --out_dim 1024 --gpu 1
#74.28+0.64
#python main.py --dataname pubmed --epochs 300 --dfr 0.3 --der 0.5 --lr2 1e-2 --wd2 1e-4 --num_groups 32 --gpu 1
#81.86+0.50

#bn
#cora 83.19+0.34 79.37+0.57
#cite 

#oversmooth
          4           8            16 
cora
cite 69.64+1.84   39.43
pub  73.73+2.35   67.67+1.76   59.43+1.31



Citeseer
 68.88 73.99 73.66 73.50 73.93 74.16 73.63 74.03
 73.60 74.13 73.68 73.95 74.02 74.00 73.91 73.78
 73.60 73.64 73.62 73.73 73.78 73.68 73.91 73.25
 74.00 73.26 73.88 73.70 73.96 74.06 73.79 74.11
 73.92 73.81 74.09 73.85 73.85 73.89 73.91 73.76
 73.95 73.70 74.03 73.46 74.02 74.15 73.51 74.16
 74.35 73.99 73.54 73.74 74.28 74.09 74.07 73.93
 73.99 73.97 73.82 73.95 73.82 74.00 73.99 73.73
 