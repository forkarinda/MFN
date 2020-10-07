import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import utils
import numpy as np

class CrossmodalFusion(nn.Module):
 
    def __init__(self, ctx1_dim, ctx2_dim):

        super().__init__()
        
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU(True)

        self.crossmodal_attention_a2v = torch.nn.Linear(ctx2_dim, ctx1_dim, bias=False) 
        self.memory_v = torch.nn.Linear(ctx1_dim, ctx1_dim)
        self.forget_v = torch.nn.Linear(ctx1_dim+ctx2_dim, ctx1_dim, bias=False) 
        self.fusion_a = torch.nn.Linear(ctx1_dim + ctx2_dim, ctx2_dim)
        
        self.crossmodal_attention_v2a = torch.nn.Linear(ctx1_dim, ctx2_dim, bias=False)  
        self.memory_a = torch.nn.Linear(ctx2_dim, ctx2_dim)
        self.forget_a = torch.nn.Linear(ctx1_dim+ctx2_dim, ctx2_dim, bias=False) 
        self.fusion_v = torch.nn.Linear(ctx1_dim + ctx2_dim, ctx1_dim)
          
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in [self.crossmodal_attention_a2v, self.crossmodal_attention_v2a, self.forget_v, self.forget_a]:
            nn.init.kaiming_normal_(layer.weight) 
        for layer in [self.fusion_a, self.fusion_v, self.memory_v, self.memory_a]:
            nn.init.kaiming_normal_(layer.weight) 
            nn.init.constant_(layer.bias, 0.)
    
    def forward(self, ctx1, ctx2):

        v = ctx1
        a = ctx2.transpose(0, 1)
       
        # text2video
        attention = torch.matmul(self.crossmodal_attention_a2v(a), v.transpose(2, 1))  # batch,n_ctx,vedio sequence
        attention_softmax=nn.Softmax(dim=-1)(attention/8)  # batch,n_ctx,vedio sequence
        v_attention = torch.matmul(attention_softmax, v)  # batch,n_ctx,vedio feature
        memory_v = self.memory_v(v_attention)
        forget_v = self.sigmoid(self.forget_v(torch.cat((a,v_attention),dim=-1)))
        v_ffg = torch.mul(memory_v,forget_v)
        #v_ffg = v_attention 
        
        
        a2v = self.fusion_a(torch.cat((a, v_ffg), dim=-1))  # batch,n_ctx,nx
        a = self.relu(a + a2v)
      
        # video2text
        attention = torch.matmul(self.crossmodal_attention_v2a(v), a.transpose(2, 1))  # batch,n_ctx,vedio sequence
        attention_softmax = nn.Softmax(dim=-1)(attention / 8)  # batch,n_ctx,vedio sequence
        a_attention = torch.matmul(attention_softmax, a)  # batch,n_ctx,vedio feature
        memory_a = self.memory_a(a_attention)
        forget_a = self.sigmoid(self.forget_a(torch.cat((v,a_attention),dim=-1)))
        a_ffg = torch.mul(memory_a, forget_a)
        #a_ffg = a_attention

        v2a = self.fusion_v(torch.cat((v, a_ffg), dim=-1))  # batch,n_ctx,nx
        v = self.relu(v + v2a)
        
        return v, a.transpose(0,1)
