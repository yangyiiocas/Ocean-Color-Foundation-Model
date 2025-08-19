import torch
import torch.nn as nn
import torch.nn.functional as F 
import math

from FNO import SpectralConv1d

def ff(in_num,out_num,hidden):
    inner_num = 64
    if hidden == 0:
        return nn.Linear(in_num,out_num)
    layer = [nn.Linear(in_num,inner_num),nn.CELU(inplace=True)]
    for i in range(hidden-1):
        layer.append(nn.Linear(inner_num,inner_num))
        layer.append(nn.CELU(inplace=True))
    layer.append(nn.Linear(inner_num,out_num))
    return nn.Sequential(*layer)


class mh_attn(nn.Module):
    def __init__(self, seq_len, d_model, heads, qkv_bias=True):
        super(mh_attn, self).__init__()
        self.heads = heads 
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((heads, 1, 1))), requires_grad=True)
        self.rpe = nn.Parameter(torch.zeros(seq_len*2-1, heads), requires_grad=True)
        x, y = torch.meshgrid(torch.arange(seq_len),torch.arange(seq_len))
        xy = x - y + seq_len - 1
        self.rpe_index = xy.flatten()
        self.fn = nn.ModuleList([nn.Linear(d_model,d_model),nn.Linear(d_model,d_model),nn.Linear(d_model,d_model)])

        self.proj = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key=None, value=None, mask=None):
        N, seq_len, d_model = query.shape
        if key is None:
            key = query
        if value is None:
            value = key
        query, key, value = [fni(x).reshape(N, seq_len, -1, self.heads).permute(0,3,1,2) for (fni,x) in zip(self.fn, (query, key, value))]     
        # cosine attention
        attn = (query @ key.transpose(-2, -1)) / math.sqrt(d_model)
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01)).to(self.logit_scale.device)).exp()
        attn = attn * logit_scale

        rpe = self.rpe[self.rpe_index, :].view(seq_len, seq_len, -1)  # seq, seq, head
        rpe = rpe.permute(2, 0, 1).contiguous()  # head, seq, seq
        rpe = torch.sigmoid(rpe)
        attn = attn + rpe.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn + mask.unsqueeze(1)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        x = (attn @ value).transpose(1, 2).reshape(N, seq_len, d_model)
        x = self.proj(x)
        return x


class sEncoder(nn.Module):
    def __init__(self,seq_len,d_model,heads):
        super(sEncoder, self).__init__()
        self.seq_len = seq_len 
        self.d_model = d_model
        self.head = heads

        self.attn1 = mh_attn(seq_len,d_model,heads)
        self.bn1 = nn.Identity() # nn.LayerNorm(normalized_shape=d_model)
        self.ffn = ff(d_model,d_model,1)
        self.fno = SpectralConv1d(d_model, d_model, seq_len//2+1)
        self.elu = nn.CELU(inplace=True)
        self.bn2 = nn.Identity() # nn.LayerNorm(normalized_shape=d_model)
    def forward(self, x, mask=None):
        x = x + self.attn1(query=x.view(-1,self.seq_len,self.d_model), mask=mask)
        x = self.elu(self.bn1(x))
        x = x + self.ffn(x) +self.fno(x.transpose(1,2)).transpose(1,2)
        return self.elu(self.bn2(x))


class sDecoder(nn.Module):
    def __init__(self,seq_len,d_model,heads):
        super(sDecoder, self).__init__()
        self.seq_len = seq_len 
        self.d_model = d_model
        self.head = heads

        self.attn1 = mh_attn(seq_len,d_model,heads)
        self.bn1 = nn.Identity() # nn.LayerNorm(normalized_shape=d_model)
        self.attn2 = mh_attn(seq_len,d_model,heads)
        self.bn2 = nn.Identity() # nn.LayerNorm(normalized_shape=d_model)
        self.fno = SpectralConv1d(d_model, d_model, seq_len//2+1)
        self.ffn = ff(d_model,d_model,3)

        self.elu = nn.CELU(inplace=True)
    def forward(self, xq, xkv, mask=None):
        x = xq + self.attn1(query=xq.view(-1,self.seq_len,self.d_model), mask=mask)
        x = self.elu(self.bn1(x))
        x = x + self.attn2(query=x.view(-1,self.seq_len,self.d_model),key=xkv.view(-1,self.seq_len,self.d_model), mask=mask)
        x = self.elu(self.bn2(x))
        x = x + self.ffn(x) + self.fno(x.transpose(1,2)).transpose(1,2)
        return x


    


if __name__ == '__main__':
    a = mh_attn(30,256,4)
    print(a(torch.rand(2,30,256),mask=torch.rand(2,30,30)).shape)
    