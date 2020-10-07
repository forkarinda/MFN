# -*- coding: utf-8 -*-
import torch.nn.functional as F

from ...utils.nn import get_rnn_hidden_state
from ..attention import HierarchicalAttention, UniformAttention, get_attention
from .. import Fusion
from . import ConditionalDecoder
import torch
import torch.nn as nn
import random
import numpy as np

class MFDecoder(ConditionalDecoder):
  
    def __init__(self, fusion_type='concat', fusion_activ=None,
                 aux_ctx_name='image', mm_att_type='md-dd',
                 persistent_dump=False, **kwargs):
        super().__init__(**kwargs)
        self.aux_ctx_name = aux_ctx_name
        self.mm_att_type = mm_att_type
        self.persistent_dump = persistent_dump

        if self.mm_att_type == 'uniform':
            # Dummy uniform attention
            self.shared_dec_state = False
            self.shared_att_mlp = False
        else:
            # Parse attention type
            att_str = sorted(self.mm_att_type.lower().split('-'))
            assert len(att_str) == 2 and att_str[0][0] == 'd' and att_str[1][0] == 'm', \
                "att_type should be m[d|i]-d[d-i]"
            # Independent <d>ecoder state means shared dec state
            self.shared_dec_state = att_str[0][1] == 'i'

            # Independent <m>odality means sharing the mlp in the MLP attention
            self.shared_att_mlp = att_str[1][1] == 'i'

            # Sanity check
            if self.shared_att_mlp and self.att_type != 'mlp':
                raise Exception("Shared attention requires MLP attention.")

        # Define (context) fusion operator
        self.fusion_type = fusion_type
        if fusion_type == "hierarchical":
            self.fusion = HierarchicalAttention(
                [self.hidden_size, self.hidden_size],
                self.hidden_size, self.hidden_size)
        else:
            if self.att_ctx2hid:
                # Old behaviour
                fusion_inp_size = 2 * self.hidden_size
            else:
                fusion_inp_sizes = list(self.ctx_size_dict.values())
                if fusion_type == 'concat':
                    fusion_inp_size = sum(fusion_inp_sizes)
                else:
                    fusion_inp_size = fusion_inp_sizes[0]
            self.fusion = Fusion(
                fusion_type, fusion_inp_size, self.hidden_size,
                fusion_activ=fusion_activ)

        # Rename textual attention layer
        self.txt_att = self.att
        del self.att

        if self.mm_att_type == 'uniform':
            self.img_att = UniformAttention()
        else:
            # Visual attention over convolutional feature maps
            Attention = get_attention(self.att_type)
            self.img_att = Attention(
                self.ctx_size_dict[self.aux_ctx_name], self.hidden_size,
                transform_ctx=self.transform_ctx, mlp_bias=self.mlp_bias,
                ctx2hid=self.att_ctx2hid,
                att_activ=self.att_activ,
                att_bottleneck=self.att_bottleneck)

        # Tune multimodal attention type
        if self.shared_att_mlp:
            # Modality independent
            self.txt_att.mlp.weight = self.img_att.mlp.weight
            self.txt_att.ctx2ctx.weight = self.img_att.ctx2ctx.weight

        if self.shared_dec_state:
            # Decoder independent
            self.txt_att.hid2ctx.weight = self.img_att.hid2ctx.weight
        
        # cross fusion
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU(True)

        ctx1_dim, ctx2_dim = 2048, 512
        
        self.crossmodal_attention_a2v = torch.nn.Linear(ctx2_dim, ctx1_dim, bias=False)
        self.memory_v = torch.nn.Linear(ctx1_dim, ctx1_dim)
        self.forget_v = torch.nn.Linear(ctx1_dim+ctx2_dim, ctx1_dim, bias=False)
        self.fusion_a = torch.nn.Linear(ctx1_dim + ctx2_dim, ctx2_dim)

        self.crossmodal_attention_v2a = torch.nn.Linear(ctx1_dim, ctx2_dim, bias=False)
        self.memory_a = torch.nn.Linear(ctx2_dim, ctx2_dim)
        self.forget_a = torch.nn.Linear(ctx1_dim+ctx2_dim, ctx2_dim, bias=False)
        self.fusion_v = torch.nn.Linear(ctx1_dim + ctx2_dim, ctx1_dim)
        
        #self.vpe = nn.Embedding(2800, 2048)
        
    def ctx_dict(self,ctx_dict):
        
        """Crossmodal Fusion"""
    
        v, v_m = ctx_dict['image'][0].transpose(0, 1), ctx_dict['image'][1]
        a, a_m = ctx_dict['tran'][0].transpose(0, 1), ctx_dict['tran'][1]
        
#         ##add video position embedding
#         vedio_position_ids = torch.arange(v.size(-2), dtype=torch.long, device=self.device)
#         vedio_position_ids = vedio_position_ids.unsqueeze(0).repeat(v.size(0),1)
#         vedio_position_embeds = self.vpe(vedio_position_ids)
#         v = v + vedio_position_embeds
 

        # t2v
        attention = torch.matmul(self.crossmodal_attention_a2v(a), v.transpose(2, 1))  # batch,n_ctx,vedio sequence
        attention_softmax=nn.Softmax(dim=-1)(attention/8)  # batch,n_ctx,vedio sequence
        v_attention = torch.matmul(attention_softmax, v)  # batch,n_ctx,vedio feature2048
        memory_v = self.memory_v(v_attention)
        forget_v = self.sigmoid(self.forget_v(torch.cat((a,v_attention),dim=-1)))
        v_ffg = torch.mul(memory_v,forget_v)
        
        a2v = self.fusion_a(torch.cat((a, v_ffg), dim=-1))  # batch,n_ctx,nx
        a = self.relu(a + a2v)
        

        # v2t
        attention = torch.matmul(self.crossmodal_attention_v2a(v), a.transpose(2, 1))  # batch,n_ctx,vedio sequence
        attention_softmax = nn.Softmax(dim=-1)(attention/8)  # batch,n_ctx,vedio sequence
        a_attention = torch.matmul(attention_softmax, a)  # batch,n_ctx,vedio feature2048
        memory_a = self.memory_a(a_attention)
        forget_a = self.sigmoid(self.forget_a(torch.cat((v,a_attention),dim=-1)))
        a_ffg = torch.mul(memory_a, forget_a)
  

        v2a = self.fusion_v(torch.cat((v, a_ffg), dim=-1))  # batch,n_ctx,nx
        v = self.relu(v + v2a)
  
        ctx_dict = {'image':(v.transpose(0,1), v_m), 'tran':(a.transpose(0,1), a_m)}
        
        return ctx_dict

    def f_next(self, ctx_dict, y, h):
        
        """Hierarchical Fusion"""
        
        # Get hidden states from the first decoder (purely cond. on LM)
        h1_c1 = self.dec0(y, self._rnn_unpack_states(h))
        h1 = get_rnn_hidden_state(h1_c1)

        # Apply attention
        self.txt_alpha_t, txt_z_t = self.txt_att(
            h1.unsqueeze(0), *ctx_dict[self.ctx_name])
        self.img_alpha_t, img_z_t = self.img_att(
            h1.unsqueeze(0), *ctx_dict[self.aux_ctx_name])
        
   
        # Save for reg loss terms
        self.history['alpha_img'].append(self.img_alpha_t.unsqueeze(0))

        # Context will double dimensionality if fusion_type is concat
        # z_t should be compatible with hidden_size
        if self.fusion_type == "hierarchical":
            self.h_att, z_t = self.fusion([txt_z_t, img_z_t], h1.unsqueeze(0))
        else:
            z_t = self.fusion(txt_z_t, img_z_t)

        if not self.training and self.persistent_dump:
            # For test-time activation debugging
            self.persistence['z_t'].append(z_t.t().cpu().numpy())
            self.persistence['txt_z_t'].append(txt_z_t.t().cpu().numpy())
            self.persistence['img_z_t'].append(img_z_t.t().cpu().numpy())

        # Run second decoder (h1 is compatible now as it was returned by GRU)
        h2_c2 = self.dec1(z_t, h1_c1)
        h2 = get_rnn_hidden_state(h2_c2)

        # This is a bottleneck to avoid going from H to V directly
        logit = self.hid2out(self.out_merge_fn(h2, y, z_t))

        # Apply dropout if any
        if self.dropout_out > 0:
            logit = self.do_out(logit)

        # Transform logit to T*B*V (V: vocab_size)
        # Compute log_softmax over token dim
        log_p = F.log_softmax(self.out2prob(logit), dim=-1)

        # Return log probs and new hidden states
        return log_p, self._rnn_pack_states(h2_c2)#, self.txt_alpha_t, self.img_alpha_t, self.h_att
    
    def forward(self, ctx_dict, y):

        ctx_dict = self.ctx_dict(ctx_dict)

        loss = 0.0

        # Get initial hidden state
        h = self.f_init(ctx_dict)

        # are we doing scheduled sampling?
        sched = self.training and (random.random() > (1 - self.sched_sample))

        # Convert token indices to embeddings -> T*B*E
        # Skip <bos> now

        bos = self.get_emb(y[0], 0)
        log_p, h = self.f_next(ctx_dict, bos, h)
        loss += self.nll_loss(log_p, y[1])
        y_emb = self.get_emb(y[1:])

        for t in range(y_emb.shape[0] - 1):
            emb = self.emb(log_p.argmax(1)) if sched else y_emb[t]
            log_p, h = self.f_next(ctx_dict, emb, h)
            loss += self.nll_loss(log_p, y[t + 2])
        
        return {'loss': loss}