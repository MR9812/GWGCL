import torch
import torch.nn as nn
from torch.nn import Parameter
import pickle
import math
linalg_device = 'cpu'


class DBN(nn.Module):
    def __init__(self, num_features, num_groups=32, num_channels=2, dim=2, eps=1e-5, momentum=0.1, affine=True, mode=0):
        super(DBN, self).__init__()
        if num_channels > 0:
            num_groups = num_features // num_channels
        self.num_features = num_features
        self.num_groups = num_groups
        assert self.num_features % self.num_groups == 0
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.mode = mode

        self.shape = [1] * dim
        self.shape[1] = num_features

        if self.affine:
            self.weight = Parameter(torch.Tensor(*self.shape))
            self.bias = Parameter(torch.Tensor(*self.shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_groups, 1))
        self.register_buffer('running_projection', torch.eye(num_groups))
        self.reset_parameters()

    # def reset_running_stats(self):
    #     self.running_mean.zero_()
    #     self.running_var.eye_(1)

    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            nn.init.uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor, shuffle=True):

        if isinstance(shuffle, list):
            input = input[shuffle[0]]


        size = input.size()
        #import ipdb;ipdb.set_trace()
        assert input.dim() == self.dim and size[1] == self.num_features
        # breakpoint()
        x = input.view(size[0] * size[1] // self.num_groups, self.num_groups, *size[2:])
        training = self.mode > 0 or (self.mode == 0 and self.training)
        x = x.transpose(0, 1).contiguous().view(self.num_groups, -1)
        # print(x.size())
        if training:
            mean = x.mean(1, keepdim=True)
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean
            x_mean = x - mean
            # sigma = x_mean.matmul(x_mean.t()) / x.size(1) + self.eps * torch.eye(self.num_groups, device=input.device)
            # print(sigma.mean())
            # with open('/root/input.pkl', 'wb+') as f:
            #     pickle.dump(x.detach().cpu().numpy(), f)
            # with open('/root/sigma.pkl', 'wb+') as f:
            #     pickle.dump(sigma.detach().cpu().numpy(), f)
            # breakpoint()
            sigma = x_mean.matmul(x_mean.t()) / x.size(1) #+ self.eps * torch.eye(self.num_groups, device=input.device)
            # print('sigma size {}'.format(sigma.size()))
            # print(sigma.shape)
            try:
                u, eig, tmp = sigma.cpu().svd()
                # import pdb
                # pdb.set_trace()
            except RuntimeError:
                import pdb
                pdb.set_trace()
                print(sigma[:5,:5])
                exit()
            # with open('/root/input.pkl', 'wb+') as f:
            #     pickle.dump(x.detach().cpu().numpy(), f)


            u = u.to(input.device)
            eig = eig.to(input.device)
            self.eig = eig[-1]
            scale = eig.rsqrt()
            # scale = 1 / eig
            wm = u.matmul(scale.diag()).matmul(u.t())
            self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * wm
            y = wm.matmul(x_mean)

            # x_mean = x - self.running_mean.detach()
            # y = self.running_projection.detach().matmul(x_mean)
        else:
            x_mean = x - self.running_mean
            y = self.running_projection.matmul(x_mean)
        output = y.view(self.num_groups, size[0] * size[1] // self.num_groups, *size[2:]).transpose(0, 1)
        output = output.contiguous().view_as(input)
        # if self.affine:
        #     output = output * self.weight + self.bias
        return output

    def extra_repr(self):
        return '{num_features}, num_groups={num_groups}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'mode={mode}'.format(**self.__dict__)


class DBN2(DBN):
    """
    when evaluation phase, sigma using running average.
    """

    def forward(self, input: torch.Tensor):
        size = input.size()
        assert input.dim() == self.dim and size[1] == self.num_features
        x = input.view(size[0] * size[1] // self.num_groups, self.num_groups, *size[2:])
        training = self.mode > 0 or (self.mode == 0 and self.training)
        x = x.transpose(0, 1).contiguous().view(self.num_groups, -1)
        mean = x.mean(1, keepdim=True) if training else self.running_mean
        x_mean = x - mean
        if training:
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean
            sigma = x_mean.matmul(x_mean.t()) / x.size(1) + self.eps * torch.eye(self.num_groups, device=input.device)
            self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * sigma
        else:
            sigma = self.running_projection
        u, eig, _ = sigma.svd()
        scale = eig.rsqrt()
        wm = u.matmul(scale.diag()).matmul(u.t())
        y = wm.matmul(x_mean)
        output = y.view(self.num_groups, size[0] * size[1] // self.num_groups, *size[2:]).transpose(0, 1)
        output = output.contiguous().view_as(input)
        if self.affine:
            output = output * self.weight + self.bias
        return output


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
        covs = x.transpose(1,2).bmm(x) / (x.size(1) - 1) # G, D//G, D//G, covs为协方差矩阵
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
    

class ShuffledGroupWhitening(nn.Module):
    def __init__(self, num_features, num_groups=None, shuffle=True, engine='symeig'):
        super(ShuffledGroupWhitening, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        if self.num_groups is not None:
            assert self.num_features % self.num_groups == 0
        # self.momentum = momentum
        self.register_buffer('running_mean', None)
        self.register_buffer('running_covariance', None)
        self.shuffle = shuffle if self.num_groups != 1 else False
        self.engine = engine

    def forward(self, x):
        N, D = x.shape
        if self.num_groups is None:
            G = math.ceil(2*D/N) # automatic, the grouped dimension 'D/G' should be half of the batch size N
            # print(G, D, N)
        else:
            G = self.num_groups
        if self.shuffle:
            new_idx = torch.randperm(D)
            x = x.t()[new_idx].t()
        x = x.view(N, G, D//G)
        x = (x - x.mean(dim=0, keepdim=True)).transpose(0,1) # G, N, D//G
        # covs = x.transpose(1,2).bmm(x) / (N-1) #  G, D//G, N @ G, N, D//G -> G, D//G, D//G
        covs = x.transpose(1,2).bmm(x) / N
        W = transformation(covs, x.device, engine=self.engine)
        x = x.bmm(W)
        if self.shuffle:
            return x.transpose(1,2).flatten(0,1)[torch.argsort(new_idx)].t()
        else:
            return x.transpose(0,1).flatten(1)

    
    
def get_corrcoef(x):
    if type(x) is torch.Tensor:
        x = x.detach().cpu().numpy()
    corr_mat = np.corrcoef(x, rowvar=False)
    np.fill_diagonal(corr_mat, 0)
    return np.abs(corr_mat).mean()

class DecorrelatedNorm(nn.Module):
    def __init__(self, num_features, memory_size=None, momentum=0.01, eps=0, memory_bank=False, mode='svd'):
        super(DecorrelatedNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(self.num_features))
        self.register_buffer("running_covariance", torch.eye(self.num_features))
        self.eps = eps
        self.memory_size = self.num_features*2 if memory_size is None else memory_size
        self.register_buffer("memory", torch.randn(self.memory_size, self.num_features) if memory_bank else None)
        self.register_buffer("memory_ptr", torch.zeros(1, dtype=torch.long) if memory_bank else None)
        self.memory_bank = memory_bank
        self.mode = mode

    @torch.no_grad()
    def recall(self, batch):
        N, M, P = batch.size(0), self.memory.size(0), int(self.memory_ptr)
        if N > M:
            print("Hey, your batch size is larger than the memory!")
            raise NotImplementedError
            self.memory = batch[N-M:N].detach()
            self.memory_ptr = 0
        elif N <= M:
            if P + N < M:
                self.memory[P:P+N] = batch.detach()
                self.memory_ptr = torch.tensor(P + N).to(self.memory_ptr)
                idx = list(range(P,P+N))
            elif P < M <= P + N:
                self.memory[P:M] = batch[:M-P].detach()
                self.memory[:P+N-M] = batch[M-P:N].detach()
                self.memory_ptr = torch.tensor(P + N - M).to(self.memory_ptr)
                idx = list(range(P,M)) + list(range(0,P+N-M))
            else:
                raise Exception
        else:
            raise Exception
        rest_idx = list(range(M))
        for i in idx:
            rest_idx.remove(i)
        return torch.cat([batch, self.memory[rest_idx].detach()])

    def forward(self, x):
        N = x.shape[0]
        if self.training:
            if self.memory_bank:
                x = self.recall(x)

            mean = x.mean(dim=0)
            x = x - mean
            cov = x.t().matmul(x) / (x.size(0) - 1)
            self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            self.running_covariance = self.momentum * cov + (1 - self.momentum) * self.running_covariance
        else:
            raise NotImplementedError
            mean = self.running_mean
            cov = self.running_covariance
            x = x - mean


        I = torch.eye(self.num_features).to(cov)
        cov = (1 - self.eps) * cov + self.eps * I

        if self.mode.startswith('svd'): # zca whitening
            if self.mode == 'svd_lowrank':
                U, S, _ = torch.svd_lowrank(cov.cpu())
            elif self.mode == 'svd':
                U, S, _ = cov.cpu().svd()
            U, S = U.to(x.device), S.to(x.device)
            whitening_transform = U.matmul(S.rsqrt().diag()).matmul(U.t())

        elif self.mode == 'cholesky':
            C = torch.cholesky(cov.cpu())
            whitening_transform = torch.triangular_solve(I.cpu(), C, upper=False)[0].t().to(x.device)

        elif self.mode.startswith('pca'):
            if self.mode == 'pca_lowrank':
                U, S, _ = torch.pca_lowrank(cov.cpu(), center=False)
                S, U = S.to(x.device), U.to(x.device)
            elif self.mode == 'pca':
                # eigenvalues, eigenvectors = torch.eig(cov.cpu(), eigenvectors=True) # S: the first element is the real part and the second element is the imaginary part, not necessary ordered
                eigenvalues, eigenvectors = torch.symeig(cov.cpu(), eigenvectors=True, upper=True)
                S, U = eigenvalues.to(x.device), eigenvectors.to(x.device)
                self.eig = eigenvalues.min()
                # breakpoint()
            whitening_transform = U.matmul(S.rsqrt().diag()).matmul(U.t())

        return x.matmul(whitening_transform)



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
        
        
        
        
        