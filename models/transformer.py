import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class MSA(nn.Module):

    def __init__(self, N, D, h):
        # N = nbr of patches
        # D = patch input dim
        # h = nbr of self attention heads
        super(MSA, self).__init__()
        self.h = h
        self.Dh = int(D / h)  # dimension of query, key & value vectors
        self.qkv = nn.Linear(N*D, 3*h*Dh, bias=False)
        self.aggregate_heads = nn.Linear(h*Dh, D)

    def __forward__(self, z):
        # z = input of dim N x 
        h, Dh = self.h, self.Dh
        Q, K, V = self.qkv(z).view(
            -1, 3, self.h, self.Dh).transpose(1, 2, 0, 3) # Q, K, V dim = h x N x Dh
        ipds = torch.matmul(Q, K.transpose(0,2,1)) / sqrt(self.Dh)  # h x N x N
        A = F.softmax(ipds, dim=2)  # weights: h x N x N
        SA = torch.matmul(A, V)  # h x N x Dh
        return self.aggregate(SA.transpose(1,0,2).view(-1, h*Dh))
