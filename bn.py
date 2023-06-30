import torch
import torch.nn as nn
from torch.nn import Parameter
import pickle
import math
linalg_device = 'cpu'


class GroupWhitening1d(nn.Module):
    def __init__(self, num_features, num_groups=32, shuffle=False, momentum=0.9):
        super(GroupWhitening1d, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        assert self.num_features % self.num_groups == 0
        #self.momentum = momentum
        # self.register_buffer("running_mean", torch.zeros(self.num_features))
        self.register_buffer('running_mean', None)
        # self.register_buffer("running_covariance", torch.eye(self.num_features))
        self.register_buffer('running_covariance', None)
        self.x_last_batch = None
        self.shuffle = shuffle

    def forward(self, x):
        #import ipdb;ipdb.set_trace()
        G, N, D = self.num_groups, *x.shape
        if self.shuffle:
            new_idx = torch.randperm(x.shape[1])
            reverse_shuffle = torch.argsort(new_idx)
            x = x.t()[new_idx].t()
        x = x.view(N, G, D//G)
        x = x - x.mean(dim=0, keepdim=True)
        x = x.transpose(0,1) # G, N, D//G
        covs = x.transpose(1,2).bmm(x) / (x.size(1) - 1) 
        #eigenvalues, eigenvectors = torch.symeig(covs.cpu(), eigenvectors=True, upper=True)
        eigenvalues, eigenvectors = torch.linalg.eigh(covs.cpu())
        S, U = eigenvalues.to(x.device), eigenvectors.to(x.device)
        self.eig = eigenvalues.min()
        whitening_transform = U.bmm(S.rsqrt().diag_embed()).bmm(U.transpose(1,2))
        x = x.bmm(whitening_transform)
        if self.shuffle:
            return x.transpose(1,2).flatten(0,1)[reverse_shuffle].t()
        else:
            return x.transpose(0,1).flatten(1)


def transformation(covs, device, engine='symeig'):
    covs = covs.to(linalg_device)
    if engine == 'cholesky':
        C = torch.cholesky(covs.to(linalg_device))
        W = torch.triangular_solve(torch.eye(C.size(-1)).expand_as(C).to(C), C, upper=False)[0].transpose(1,2).to(x.device)
    else:
        if engine == 'symeig':
            S, U = torch.symeig(covs.to(linalg_device), eigenvectors=True, upper=True)
        elif engine == 'svd':
            U, S, _ = torch.svd(covs.to(linalg_device))
        elif engine == 'svd_lowrank':
            U, S, _ = torch.svd_lowrank(covs.to(linalg_device))
        elif engine == 'pca_lowrank':
            U, S, _ = torch.pca_lowrank(covs.to(linalg_device), center=False)
        S, U = S.to(device), U.to(device)
        W = U.bmm(S.rsqrt().diag_embed()).bmm(U.transpose(1,2))
    return W


class Whitening1d(nn.Module):
    def __init__(self, num_features, momentum=0.01, eps=1e-5):
        super(Whitening1d, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(self.num_features))
        self.register_buffer("running_covariance", torch.eye(self.num_features))
        self.eps = eps
    def forward(self, x, numpy=False):
        if self.training:
            mean = x.mean(dim=0)
            x = x - mean
            cov = x.t().matmul(x) / (x.size(0) - 1)
            self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            self.running_covariance = self.momentum * cov + (1 - self.momentum) * self.running_covariance
        else:
            mean = self.running_mean
            cov = self.running_covariance
            x = x - mean

        cov = (1 - self.eps) * cov + self.eps * torch.eye(self.num_features).to(cov)
        if numpy:

            I = torch.eye(x.size(1)).to(cov).detach().cpu().numpy()
            cv = np.linalg.cholesky(cov.detach().cpu().numpy())
            whitening_transform = solve_triangular(cv, I, lower=True).T
            
            whitening_transform = torch.tensor(whitening_transform).to(x)
        else:
            I = torch.eye(x.size(1)).to(cov).cpu()
            C = torch.cholesky(cov.cpu())
            whitening_transform = torch.triangular_solve(I, C, upper=False)[0].t().to(x.device)
        return x.matmul(whitening_transform)
        
        
        
        
        
