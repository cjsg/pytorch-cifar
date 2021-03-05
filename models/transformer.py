import torch
import torch.nn as nn
from math import sqrt
from einops import rearrange, repeat

'''
    l = number of attention layers
    d = dim of latent feature vector
    h = number of SA heads
    n = number of image patches
    p = patch height / width
    c = number of channels
    k = number of classification classes
    f = dim of qurey, key and value features
    NB: image dim = c * n * p^2

'''

class Embed(nn.Module):

    def __init__(self, d, n, p, c=3, dropout_rate=0.):
        super(Embed, self).__init__()
        self.d, self.n, self.p, self.c = d, n, p, c
        self.embed = nn.Linear(c*p**2, d)
        self.cls = nn.parameter.Parameter(torch.randn(d))
        self.pos = nn.parameter.Parameter(torch.randn(n+1, d))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = rearrange(x, 'b c (h1 h2) (w1 w2) -> b (h1 w1) (h2 w2 c)',
                      h2=self.p, w2=self.p)
        cls = repeat(self.cls, 'd -> b 1 d', b=x.size(0))
        out = torch.cat([cls, self.embed(x)], dim=1)  # b x (n+1) x d
        return self.dropout(out + self.pos)

class MSA(nn.Module):

    def __init__(self, n, d, h, dropout_rate=0.):
        # n = nbr of image patches
        # d = patch input dim
        # h = nbr of self attention heads
        super(MSA, self).__init__()
        f = int(d / h)  # dimension of query, key & value vectors ('*f*eatures')
        self.h, self.d, self.f = h, d, f
        self.to_qkv = nn.Linear(d, 3*h*f, bias=False)  # input feature to q,k,v vectors
        self.to_mlp = nn.Sequential(
            nn.Linear(h*f, d),  # aggregate attention heads and send to MLP
            nn.Dropout(dropout_rate),)

    def forward(self, z):
        b, n, d, h, f = *z.shape, self.h, self.f  # -> renaming n := n+1
        qkv = self.to_qkv(z)
        q, k, v = rearrange(qkv, 'b n (i h f) -> i b h n f', i=3, h=h)  # 3 x b h n f
        dots = torch.einsum('bhnf,bhmf->bhnm', q, k) / sqrt(f) # here, n=m=n
        attn = dots.softmax(dim=3)  # attention weights: b h n n
        slf_attn = torch.einsum('bhnm,bhmf->bhnf', attn, v)  # b h n f
        slf_attn = rearrange(slf_attn, 'b h n f -> b n (h f)')
        return self.to_mlp(slf_attn)  # b n d


class MLP(nn.Module):

    def __init__(self, d, dropout_rate=0.):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(d, d),
            nn.Dropout(dropout_rate),)

    def forward(self, z):
        return self.mlp(z)

class Transformer(nn.Module):
    def __init__(self, d, h, n, p, c=3, dropout_rate=0.):
        super(Transformer, self).__init__()
        self.ln1 = nn.LayerNorm([n+1, d])  # lucidrains uses LayerNorm(d)
        self.msa = MSA(n, d, h, dropout_rate)
        self.ln2 = nn.LayerNorm([n+1, d])  # lucidrains uses LayerNorm(d)
        self.mlp = MLP(d, dropout_rate)

    def forward(self, z):
        # dim(z) = b x (n+1) x d
        z = self.msa(self.ln1(z)) + z
        z = self.mlp(self.ln2(z)) + z
        return z


class ViT(nn.Module):
    def __init__(self, l, d, h, n, p, c=3, k=10, dropout_rate=0.):
        super(ViT, self).__init__()
        self.embedding = Embed(d, n, p, c)
        self.transformer = nn.Sequential(
            *[Transformer(d, h, n, p, c, dropout_rate) for _ in range(l)])
        self.ln = nn.LayerNorm(d)
        self.to_class_logits = nn.Linear(d, k)

    def forward(self, x):
        z = self.embedding(x)
        z = self.transformer(z)
        return self.to_class_logits(self.ln(z[:,0]))


def ViT_L2_H4_P4(dropout_rate=0.):
    return ViT(
        l=2, h=4, n=8*8, d=4*4*3, p=4, c=3, k=10, dropout_rate=dropout_rate)

def ViT_L8_H4_P4(dropout_rate=0.):
    return ViT(
        l=8, h=4, n=8*8, d=4*4*3, p=4, c=3, k=10, dropout_rate=dropout_rate)


def test():
    net = Transformer_L8_H4_P4()
    y = net(torch.randn(1, 3, 32, 32))
    print(y)
    print(y.size())

# test()
