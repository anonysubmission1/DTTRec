import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, attn_mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class GITA_ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, d, seq_len, attn_dropout=0.1):
        super(GITA_ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.layer_norm1 = nn.LayerNorm(seq_len)
        self.layer_norm2 = nn.LayerNorm(seq_len)
        self.mlp1 = nn.Linear(d*2, d*2)
        self.mlp2 = nn.Linear(d*2, 1)
        nn.init.normal_(self.mlp1.weight, mean=0, std=np.sqrt(2.0 / (d*2 + d*2)))
        nn.init.zeros_(self.mlp2.weight)
        nn.init.zeros_(self.mlp1.bias)
        nn.init.constant_(self.mlp2.bias, val=-4.0)

    def forward(self, q, k, v, q2, k2, attn_mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        attn2 = torch.bmm(q2, k2.transpose(1, 2))
        attn2 = attn2 / self.temperature

        gate = torch.sigmoid(self.mlp2(torch.relu(self.mlp1(torch.cat([q, q2], dim=-1))))) # b * l * 1
        attn2 = gate * attn2
        
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -np.inf)
            attn2 = attn2.masked_fill(attn_mask, -np.inf)

        attn = self.softmax(attn + attn2)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)


    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        attn_mask = attn_mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, attn_mask=attn_mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        return output, attn


class GITA_MultiHeadAttention(nn.Module):
    def __init__(self, n_head, seq_len, d_model, d_k, d_v, dropout=0.1):
        super(GITA_MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.w_qs2 = nn.Linear(d_model, n_head * d_k)
        self.w_ks2 = nn.Linear(d_model, n_head * d_k)
        nn.init.normal_(self.w_qs2.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks2.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))

        self.attention = GITA_ScaledDotProductAttention(temperature=np.power(d_k, 0.5), d=d_model, seq_len=seq_len, attn_dropout=dropout)


    def forward(self, q, k, v, q2, k2, attn_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q2 = self.w_qs2(q2).view(sz_b, len_q, n_head, d_k)
        k2 = self.w_ks2(k2).view(sz_b, len_k, n_head, d_k)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        q2 = q2.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k2 = k2.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        
        attn_mask = attn_mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, q2, k2, attn_mask=attn_mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) 
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(self.dropout(F.relu(self.w_1(output))))
        output = output.transpose(1, 2)
        return output


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(SelfAttentionBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        residual = enc_input
        enc_input = self.layer_norm1(enc_input)

        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        enc_output = residual + self.dropout(enc_output)
        enc_output *= non_pad_mask

        residual = enc_output
        enc_output = self.layer_norm2(enc_output)

        enc_output = self.pos_ffn(enc_output)
        enc_output = residual + self.dropout(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class AttentionBlock(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.attn = torch.nn.MultiheadAttention(d_model, n_head, dropout, batch_first=True)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.layer_norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, non_pad_mask=None, attn_mask=None):
        residual = Q
        Q = self.layer_norm1(Q)
        K = self.layer_norm2(K)
        V = self.layer_norm3(V)

        enc_output, enc_slf_attn = self.attn(Q, K, V, attn_mask=attn_mask)
        enc_output = residual + self.dropout(enc_output)
        enc_output *= non_pad_mask

        residual = enc_output
        enc_output = self.layer_norm4(enc_output)

        enc_output = self.pos_ffn(enc_output)
        enc_output = residual + self.dropout(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class GITABlock(nn.Module):
    def __init__(self, d_model, d_inner, seq_len, n_head, d_k, d_v, dropout=0.1):
        super(GITABlock, self).__init__()
        self.attn = GITA_MultiHeadAttention(n_head, seq_len, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.layer_norm1_Q = nn.LayerNorm(d_model)
        self.layer_norm1_K = nn.LayerNorm(d_model)
        self.layer_norm1_V = nn.LayerNorm(d_model)
        self.layer_norm2_Q = nn.LayerNorm(d_model)
        self.layer_norm2_K = nn.LayerNorm(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, Q2, K2, non_pad_mask=None, attn_mask=None):
        residual = Q
        Q = self.layer_norm1_Q(Q)
        K = self.layer_norm1_K(K)
        V = self.layer_norm1_V(V)
        Q2 = self.layer_norm2_Q(Q2)
        K2 = self.layer_norm2_K(K2)

        enc_output, enc_slf_attn = self.attn(Q, K, V, Q2, K2, attn_mask=attn_mask)
        enc_output = residual + self.dropout(enc_output)
        enc_output *= non_pad_mask

        residual = enc_output
        enc_output = self.layer_norm(enc_output)

        enc_output = self.pos_ffn(enc_output)
        enc_output = residual + self.dropout(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn
