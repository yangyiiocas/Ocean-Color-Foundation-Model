import torch
import torch.nn as nn
import torch.nn.functional as F 
from FNO import SpectralConv1d
from MHAttn import sDecoder, sEncoder

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Attention(nn.Module):
    def __init__(self, d_model, dim_qkv):
        super(Attention, self).__init__()
        self.dim_qkv = dim_qkv
        self.fn = nn.ModuleList([nn.Linear(d_model, dim_qkv[0]), nn.Linear(d_model, dim_qkv[0]), nn.Linear(d_model, dim_qkv[1])])
    def forward(self, x):
        query, key, value = [fni(x) for fni in self.fn]
        attn_scores = torch.bmm(query, key.transpose(1, 2)) / float(self.dim_qkv[0]**0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        return torch.bmm(attn_probs, value)


class InputEmbedding(nn.Module):
    def __init__(self, Lx, seq_len, d_model):
        super(InputEmbedding, self).__init__()
        self.fno = SpectralConv1d(Lx, d_model, seq_len//2+1)
        self.fn = nn.Linear(Lx, d_model)
    def forward(self, x):
        # Input Size -> (N, Lx, seq_len)
        # Output Size -> (N, seq_len, d_model)
        x = self.fno(x).transpose(1,2) + self.fn(x.transpose(1,2))
        return x


class OutputDeEmbedding(nn.Module):
    def __init__(self, seq_len, d_model, k=8, dim_qkv=(16, 16)):
        super(OutputDeEmbedding, self).__init__()
        self.proj = nn.Conv1d(1,1,k,stride=k)
        self.attn = Attention(d_model//k, dim_qkv)
        self.fn = nn.Linear(d_model//k, dim_qkv[1])
        self.fn_out = nn.Sequential(nn.Linear(dim_qkv[1],4*dim_qkv[1]),
                                    nn.PReLU(),
                                    nn.Linear(4*dim_qkv[1],4*dim_qkv[1]),
                                    nn.PReLU(),
                                    nn.Linear(4*dim_qkv[1],1))
        self.elu = nn.PReLU()
    def forward(self, x):
        # Input Size -> (N, seq_len, d_model)
        # Output Size -> (N, 1)
        N, seq_len, _ = x.size()
        x = x.max(dim=1, keepdim=True).values
        x = self.proj(x)
        x = self.attn(x) + self.fn(x)
        x = self.elu(x)
        return self.fn_out(x.reshape(N,-1))

class Encoder_Decoder(nn.Module):
    def __init__(self, d_model, seq_len, encoder_num=2, decoder_num=2, heads=4):
        super(Encoder_Decoder, self).__init__()
        self.PE = nn.Parameter((torch.arange(1, seq_len+1, 1)/seq_len*torch.pi/2).sin_().unsqueeze(dim=1).repeat(1, d_model), requires_grad=False)
        self.s_encoders = nn.ModuleList(sEncoder(seq_len, d_model, heads) for i in range(encoder_num))
        self.s_decoders = nn.ModuleList(sDecoder(seq_len, d_model, heads) for i in range(decoder_num))

    def forward(self, x_kv, mask=None):
        # Input Size -> (N, seq_len, d_model)
        # Output Size -> (N, seq_len, d_model)
        N, seq_len, d_model = x_kv.size()

        x_kv = x_kv + self.PE.view(1, seq_len, d_model)
        for s_encoder in self.s_encoders:
            x_kv = s_encoder(x_kv, mask=mask)
        
        x1 = self.PE.view(1, seq_len, d_model).expand(N, seq_len,d_model)
        for i,s_decoder in enumerate(self.s_decoders):
                x1 = s_decoder(x1, x_kv, mask=mask)
        return x1

    
    
    
class model_a(nn.Module):
    def __init__(self, Lx, d_model, seq_len, tasks, encoder_num=2, decoder_num=2, heads=4, dim_k=None, dim_v=None):
        super(model_a, self).__init__()
        self.model = Encoder_Decoder(d_model=d_model, seq_len=seq_len, encoder_num=encoder_num, decoder_num=decoder_num, heads=heads)
        print(f'model parameters: {count_parameters(self.model)}')
        self.in_Embed = InputEmbedding(Lx=Lx,seq_len=seq_len,d_model=d_model)
        print(f'model parameters: {count_parameters(self.in_Embed)}')

        self.out_dEmbed = nn.ModuleDict({item:OutputDeEmbedding(seq_len=seq_len, d_model=d_model) for item in tasks})
        print(f'model parameters: {count_parameters(self.out_dEmbed)//len(self.out_dEmbed)}, {len(self.out_dEmbed)}')
        
        
    def forward(self, x, cho_v=None):
        N = x.size(0)
        ava = 1. - torch.isnan(x).any(1).unsqueeze(dim=-1).to(x.dtype)
        mask = 1. - ava @ ava.transpose(1,2)
        x = torch.nan_to_num(x, nan=0)
        out = self.model(self.in_Embed(x), mask=None)
        if cho_v is None:
            return {vari:d(out) for vari, d in self.out_dEmbed.items()}   
        else:
            return {vari:d(out) for vari, d in self.out_dEmbed.items() if vari in cho_v}

if __name__ == '__main__':
    # from sdata import tasks  
    # print(torch.cuda.is_available())
    # model = model_a(21, 256, 5, tasks)
    # input = torch.rand(1,21,5)
    # input[0,4,1] = torch.nan
    # input[0,1,1] = torch.nan
    # input[0,20,4] = torch.nan
    # a = model(input)
    
    model = OutputDeEmbedding(30,256)
    print(model(torch.rand(4,30,256)).shape, count_parameters(model))
    # print(a['chlor_a'].shape)