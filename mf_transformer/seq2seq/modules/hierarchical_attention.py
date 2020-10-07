import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import utils



class HierarchicalAttention(nn.Module):
 
    def __init__(self, ctx_dim, hid_dim, att_activ='tanh'):

        super().__init__()

        self.ctx_dim = ctx_dim
        self.hid_dim = hid_dim
        self.activ = getattr(torch, att_activ)

        self.ctx2ctx = nn.Linear(self.ctx_dim, self.hid_dim, bias=False)
        self.hid2ctx = nn.Linear(self.hid_dim, self.hid_dim, bias=False)
        self.mlp = nn.Linear(self.hid_dim, 1, bias=False)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for layer in [self.ctx2ctx, self.hid2ctx, self.mlp]:
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, hid, ctx1, ctx2, ctx_mask=None):

        ctx = torch.cat((ctx1.unsqueeze(1),ctx2.unsqueeze(1)),1)

        inner_sum = self.ctx2ctx(ctx) + self.hid2ctx(hid).unsqueeze(1)

        scores = self.mlp(
            self.activ(inner_sum)).squeeze(-1)

        if ctx_mask is not None:
            scores.masked_fill_((1 - ctx_mask).byte(), -1e8)
    
        alpha = F.softmax(scores, dim=1) 
        joint_context = (alpha.unsqueeze(-1) * ctx).sum(1)

        
        return alpha, joint_context
