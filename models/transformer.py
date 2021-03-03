import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Embed(nn.Module):

    def __init__(self, D, N, P, C=3):
        # D = latent feature vector dimension
        # N = number of image patches
        # P = patch height / width
        # C = number of channels
        # NB: image dim = C * N * P^2
        super(Embed, self).__init__()
        self.D, self.N, self.P, self.C = D, N, P, C
        self.embed = nn.Linear(C*P**2, D)
        self.pos = nn.parameter.Parameter(torch.randn(N+1, D))

    def forward(self, x):
        bs = x.size(0)
        x = x.permute(0, 2, 3, 1).reshape(bs, self.N, self.C*self.P**2)  # .view does not work
        out = torch.cat([torch.zeros(bs, 1, self.D), self.embed(x)], dim=1)  # bs x (N+1) x D
        return out + self.pos

class MSA(nn.Module):

    def __init__(self, N, D, H):
        # N = nbr of image patches
        # D = patch input dim
        # H = nbr of self attention heads
        super(MSA, self).__init__()
        Dh = int(D / H)  # dimension of query, key & value vectors
        self.H, self.D, self.Dh = H, D, Dh
        self.qkv = nn.Linear(D, H*3*Dh, bias=False)  # project input features to q,k,v vectors
        self.aggregate = nn.Linear(H*Dh, D)  # aggregate H attention heads

    def forward(self, z):
        # z = input of dim bs x (N+1) x D
        bs, N, D = z.size()  # -> renaming N := N+1
        H, Dh = self.H, self.Dh
        QKV = self.qkv(z.view(bs*N, D)).view(bs, N, 3, H, Dh)
        Q, K, V = QKV.permute(2, 0, 3, 1, 4) # dims of Q, K, V = bs x H x N x Dh
        IPDs = torch.matmul(Q, K.transpose(2, 3)) / sqrt(Dh)  # bs x H x N x N
        A = F.softmax(IPDs, dim=3)  # weights: bs x H x N x N
        SA = torch.matmul(A, V)  # bs x H x N x Dh
        SA = SA.permute(0, 2, 1, 3).reshape(bs*N, H*Dh)  # .view does not work
        return self.aggregate(SA).view(bs, N, self.D)


class MLP(nn.Module):

    def __init__(self, D):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(D, D)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(D, D)

    def forward(self, z):
        return self.layer2(self.gelu(self.layer1(z)))


class TransformerLayer(nn.Module):
    def __init__(self, D, H, N, P, C=3):
        # D = latent feature/embedding dimension
        # H = number of SA heads
        # N = number of image patches
        # P = height and wifth of an image patch
        # C = number of image channels
        # NB: dim(image) = N * C * P^2

        super(TransformerLayer, self).__init__()
        self.ln1 = nn.Identity()  # nn.LayerNorm()
        self.msa = MSA(N, D, H)
        self.ln2 = nn.Identity()  # nn.LayerNorm()
        self.mlp = MLP(D)

    def forward(self, z):
        z = self.msa(self.ln1(z)) + z
        z = self.mlp(self.ln2(z)) + z
        return z


class Transformer(nn.Module):
    def __init__(self, L, D, H, N, P, C=3, K=10):
        # L = number of attention layers
        # D = latent feature vector dimension
        # H = number of SA heads
        # N = number of image patches
        # P = patch height / width
        # C = number of channels
        # K = number of classification classes
        # NB: image dim = C * N * P^2

        super(Transformer, self).__init__()
        self.embedding = Embed(D, N, P, C)
        self.transformer = nn.Sequential(
            *[TransformerLayer(D, H, N, P, C) for _ in range(L)])
        self.ln = nn.Identity()  # nn.LayerNorm()
        self.linear = nn.Linear(D, K)

    def forward(self, x):
        z = self.embedding(x)
        z = self.transformer(z)
        return self.linear(self.ln(z[:,0]))


def Transformer_L2_H4_P4():
    return Transformer(L=2, H=4, N=8*8, D=4*4*3, P=4, C=3, K=10)

def Transformer_L8_H4_P4():
    return Transformer(L=8, H=4, N=8*8, D=4*4*3, P=4, C=3, K=10)


def test():
    net = Transformer_L8_H4_P4()
    y = net(torch.randn(1, 3, 32, 32))
    print(y)
    print(y.size())

# test()
