#group whiten
#python main.py --dataname cora --epochs 100 --dfr 0.1 --der 0.6 --lr1 5e-4 --lr2 1e-2 --wd2 1e-4 --num_groups 32 --use_norm True --gpu 0
#python main.py --dataname citeseer --epoch 50 --n_layers 1 --dfr 0.4 --der 0.6 --lr2 5e-2 --wd2 1e-2 --num_groups 16 --hid_dim 1024 --out_dim 1024 --gpu 0
#python main.py --dataname pubmed --epochs 300 --dfr 0.3 --der 0.5 --lr2 1e-2 --wd2 1e-4 --num_groups 32 --gpu 0
