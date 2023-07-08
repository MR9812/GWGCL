import argparse
from model import GWGRL, LogReg
from aug import random_aug
from dataset import load

import numpy as np
import torch as th
import torch.nn as nn
import random
import warnings
import torch.nn.functional as F

def one_hot(x, class_count):
    return th.eye(class_count)[x,:]

def main():
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='GWGRL')
    parser.add_argument('--dataname', type=str, default='cora', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id.')
    parser.add_argument("--runs", type=int, default=10, help='run times.')
    
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs.')
    parser.add_argument('--lr1', type=float, default=1e-3, help='Learning rate of GWGRL.')
    parser.add_argument('--wd1', type=float, default=0, help='Weight decay of GWGRL.')
    parser.add_argument('--lr2', type=float, default=1e-2, help='Learning rate of linear evaluator.')
    parser.add_argument('--wd2', type=float, default=1e-4, help='Weight decay of linear evaluator.')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
    
    parser.add_argument('--der', type=float, default=0.2, help='Drop edge ratio.')
    parser.add_argument('--dfr', type=float, default=0.2, help='Drop feature ratio.')
    parser.add_argument("--hid_dim", type=int, default=512, help='Hidden layer dim.')
    parser.add_argument("--out_dim", type=int, default=512, help='Output layer dim.')
    parser.add_argument('--num_groups', type=int, default=32, help='group whiten nums.')
    parser.add_argument('--use_norm', type=bool, default=False, help='embedding norm.')
    args = parser.parse_args()
    
    # check cuda
    if args.gpu != -1 and th.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'
    
    citegraph = ['cora', 'citeseer', 'pubmed']
    if args.dataname in citegraph:
        graph, feat, labels, num_class, train_idx, val_idx, test_idx = load(args.dataname)
        in_dim = feat.shape[1]
        N = graph.number_of_nodes()
    
    tests = []
    for run in range(args.runs):
        if args.dataname not in citegraph:
            graph, feat, labels, num_class, train_idx, val_idx, test_idx = load(args.dataname)
            in_dim = feat.shape[1]
            N = graph.number_of_nodes()
        
        graph = graph.cpu()
        feat = feat.cpu()
        model = GWGRL(in_dim, args.hid_dim, args.out_dim, args.n_layers, args.num_groups)
        model = model.to(args.device)
        optimizer = th.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
        
        loss_min = 1000000000
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
    
            graph1, feat1 = random_aug(graph.cpu(), feat.cpu(), args.dfr, args.der)
            graph2, feat2 = random_aug(graph.cpu(), feat.cpu(), args.dfr, args.der)
    
            graph1 = graph1.add_self_loop()
            graph2 = graph2.add_self_loop()
    
            graph1 = graph1.to(args.device)
            graph2 = graph2.to(args.device)
    
            feat1 = feat1.to(args.device)
            feat2 = feat2.to(args.device)
    
            z1 = model(graph1, feat1)
            z2 = model(graph2, feat2)
            if args.use_norm == True:
                z1 = F.normalize(z1)
                z2 = F.normalize(z2)
            loss = (z1 - z2).norm(dim=1, p=2).pow(2).mean()
    
            loss.backward()
            optimizer.step()
    
            print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))
    
        print("=== Evaluation ===")
        graph = graph.to(args.device)
        graph = graph.remove_self_loop().add_self_loop()
        feat = feat.to(args.device)
        embeds = model.get_embedding(graph, feat)
    
        train_embs = embeds[train_idx]
        val_embs = embeds[val_idx]
        test_embs = embeds[test_idx]
    
        label = labels.to(args.device)
        train_labels = label[train_idx]
        val_labels = label[val_idx]
        test_labels = label[test_idx]
    
        ''' Linear Evaluation '''
        logreg = LogReg(train_embs.shape[1], num_class)
        opt = th.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)
        
        logreg = logreg.to(args.device)
        loss_fn = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        eval_acc = 0
        
        for epoch in range(2000):
            logreg.train()
            opt.zero_grad()
            logits = logreg(train_embs)
            preds = th.argmax(logits, dim=1)
            train_acc = th.sum(preds == train_labels).float() / train_labels.shape[0]
            loss = loss_fn(logits, train_labels)
            loss.backward()
            opt.step()
        
            logreg.eval()
            with th.no_grad():
                val_logits = logreg(val_embs)
                test_logits = logreg(test_embs)
        
                val_preds = th.argmax(val_logits, dim=1)
                test_preds = th.argmax(test_logits, dim=1)
        
                val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]
                test_acc = th.sum(test_preds == test_labels).float() / test_labels.shape[0]
        
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    eval_acc = test_acc
        
                print('Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_acc:{:4f}'.format(epoch, train_acc, val_acc, test_acc))
    
        print('Linear evaluation accuracy:{:.4f}'.format(eval_acc))
        tests.append(eval_acc.cpu())
        
    print('')
    print("test:", tests)
    print(f"Average test accuracy: {np.mean(tests)} ± {np.std(tests):.6f}")
    print(args)



def set_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.manual_seed(seed)

if __name__ == "__main__":
    set_seed()
    main()








