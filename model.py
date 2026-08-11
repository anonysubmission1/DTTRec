import torch
import torch.nn as nn
import torch.nn.functional as F
from model_util import *

class DTTRec(nn.Module):
    def __init__(self, num_items, embedding_dim, max_length, num_layers, dropout, month=True, wday=True, hour=True):
        super(DTTRec, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_layers = num_layers
        self.max_length = max_length
        self.month = month
        self.wday = wday
        self.hour = hour

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.dropout_p = dropout
        self.dropout = nn.Dropout(self.dropout_p)

        self.item_embedding = nn.Embedding(self.num_items + 1, self.embedding_dim, padding_idx = 0)
        self.positional_embedding = nn.Embedding(self.max_length, self.embedding_dim)
        
        self.item_ln = nn.LayerNorm(self.embedding_dim)

        self.item_mlp1 = nn.ModuleList([nn.Linear(self.embedding_dim * 2, self.embedding_dim) for _ in range(self.num_layers)])
        self.item_mlp2 = nn.ModuleList([nn.Linear(self.embedding_dim, self.embedding_dim) for _ in range(self.num_layers)])
        self.item_mlp3 = nn.ModuleList([nn.Linear(self.embedding_dim * 2, self.embedding_dim) for _ in range(self.num_layers)])

        self.time_mlp1 = nn.ModuleList([nn.Linear(self.embedding_dim * 2, self.embedding_dim) for _ in range(self.num_layers)])
        self.time_mlp2 = nn.ModuleList([nn.Linear(self.embedding_dim, self.embedding_dim) for _ in range(self.num_layers)])
        self.time_mlp3 = nn.ModuleList([nn.Linear(self.embedding_dim * 2, self.embedding_dim) for _ in range(self.num_layers)])

        self.item_ln1 = nn.ModuleList([nn.LayerNorm(self.embedding_dim) for _ in range(self.num_layers)])
        self.item_ln2 = nn.ModuleList([nn.LayerNorm(self.embedding_dim) for _ in range(self.num_layers)])
        self.item_ln3 = nn.ModuleList([nn.LayerNorm(self.embedding_dim) for _ in range(self.num_layers)])

        self.time_ln1 = nn.ModuleList([nn.LayerNorm(self.embedding_dim) for _ in range(self.num_layers)])
        self.time_ln2 = nn.ModuleList([nn.LayerNorm(self.embedding_dim) for _ in range(self.num_layers)])
        self.time_ln3 = nn.ModuleList([nn.LayerNorm(self.embedding_dim) for _ in range(self.num_layers)])

        if self.month and self.wday and self.hour:
            self.num_time_components = 3
        elif (self.month and self.wday) or (self.month and self.hour) or (self.wday and self.hour):
            self.num_time_components = 2
        elif self.month or self.wday or self.hour: 
            self.num_time_components = 1
        else: assert False

        self.fourier_mlp1 = nn.Linear(self.num_time_components * 2, self.num_time_components * 4) 
        self.fourier_mlp2 = nn.Linear(self.num_time_components * 4, self.embedding_dim) 
        self.final_time_layer = nn.Linear(self.embedding_dim, self.num_time_components * 2) 
        
        self.log_concentration = nn.Parameter(torch.zeros(self.num_time_components))

        self.self_attention = nn.ModuleList([AttentionBlock(self.embedding_dim, self.embedding_dim, 1, self.embedding_dim, self.embedding_dim, dropout=self.dropout_p) for _ in range(self.num_layers)])
        self.item_gita = nn.ModuleList([GITABlock(self.embedding_dim, self.embedding_dim, self.max_length, 1, self.embedding_dim, self.embedding_dim, dropout=self.dropout_p) for _ in range(self.num_layers)])
        self.time_gita = nn.ModuleList([GITABlock(self.embedding_dim, self.embedding_dim, self.max_length, 1, self.embedding_dim, self.embedding_dim, dropout=self.dropout_p) for _ in range(self.num_layers)])


    def time_func(self, time_seq, item_seq, item_rep, time_mlp1, time_mlp2, time_mlp3, time_ln1, time_ln2, time_ln3):
        seq_rep = torch.cat([time_seq, item_seq], dim=-1)
        seq_rep = self.dropout(time_ln2(time_mlp2(self.dropout(torch.relu(time_ln1(time_mlp1(seq_rep))))))) 
        time_rep = self.dropout(time_mlp3(torch.cat([seq_rep, item_rep], dim=-1))) 
        time_rep = time_ln3(time_rep + time_seq)
        return time_rep 


    def item_func(self, item_seq, time_seq, time_rep, item_mlp1, item_mlp2, item_mlp3, item_ln1, item_ln2, item_ln3):
        seq_rep = torch.cat([item_seq, time_seq], dim=-1)
        seq_rep = self.dropout(item_ln2(item_mlp2(self.dropout(torch.relu(item_ln1(item_mlp1(seq_rep))))))) 
        item_rep = self.dropout(item_mlp3(torch.cat([seq_rep, time_rep], dim=-1))) 
        item_rep = item_ln3(item_rep + item_seq)
        return item_rep


    def calendar_fourier_encoding(self, months, wdays, hours, isValid):
        # b * l
        num_months = 12.0
        num_wdays = 7.0
        num_hours = 24.0
        months_norm = months / num_months
        wdays_norm = wdays / num_wdays
        hours_norm = hours / num_hours

        # b * l * 2
        month_phi = torch.cat([torch.sin(2 * torch.pi * months_norm).unsqueeze(-1), torch.cos(2 * torch.pi * months_norm).unsqueeze(-1)], dim=-1)
        wday_phi = torch.cat([torch.sin(2 * torch.pi * wdays_norm).unsqueeze(-1), torch.cos(2 * torch.pi * wdays_norm).unsqueeze(-1)], dim=-1)
        hour_phi = torch.cat([torch.sin(2 * torch.pi * hours_norm).unsqueeze(-1), torch.cos(2 * torch.pi * hours_norm).unsqueeze(-1)], dim=-1)
        
        # b * l * 2|C|
        if self.month and self.wday and self.hour:
            phi = torch.cat([month_phi, wday_phi, hour_phi], dim=-1) 
        elif self.month and self.wday:
            phi = torch.cat([month_phi, wday_phi], dim=-1) 
        elif self.month and self.hour:
            phi = torch.cat([month_phi, hour_phi], dim=-1) 
        elif self.wday and self.hour:
            phi = torch.cat([wday_phi, hour_phi], dim=-1) 
        elif self.month:
            phi = torch.cat([month_phi], dim=-1) 
        elif self.wday:
            phi = torch.cat([wday_phi], dim=-1) 
        elif self.hour:
            phi = torch.cat([hour_phi], dim=-1) 
        else: assert False
                
        return phi * isValid.unsqueeze(2)


    def ind_calendar_fourier_encoding(self, periods):
        num_period = periods.shape[0]
        periods_norm = periods / num_period 
        period_phi = torch.cat([torch.sin(2 * torch.pi * periods_norm).unsqueeze(-1), torch.cos(2 * torch.pi * periods_norm).unsqueeze(-1)], dim=-1)
        return period_phi


    def forward(self, sequences, months, wdays, hours, month_labels, wday_labels, hour_labels, test=False):
        batch_size = sequences.size()[0]
        max_length = sequences.size()[1]
        isValid = (sequences != 0.0).float()

        # Build embedding
        positionalEmbeddings = self.positional_embedding(torch.arange(self.max_length).to(self.device))

        fourierEncoding = self.calendar_fourier_encoding(months, wdays, hours, isValid)
        fourierEncoding = self.fourier_mlp2(torch.relu(self.fourier_mlp1(fourierEncoding))) 

        itemEmbeddings = self.item_embedding(sequences) + positionalEmbeddings 
        itemEmbeddings = self.item_ln(itemEmbeddings)
        itemEmbeddings = self.dropout(itemEmbeddings) 
        itemEmbeddings *= isValid.unsqueeze(2)

        # mask
        subsequent_mask = torch.triu(torch.ones((max_length, max_length), device=self.device, dtype=torch.uint8), diagonal=1)
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, -1, -1) 
        padding_mask = sequences.eq(0) # batch_size * max_length
        padding_mask = padding_mask.unsqueeze(1).expand(-1, max_length, -1) 
        mask = (subsequent_mask + padding_mask).gt(0)
        mask[:, :, 0] = 0
        non_pad_mask = sequences.ne(0).type(torch.float).unsqueeze(-1)

        next_item_rep_list = []
        next_time_rep_list = []

        for i in range(self.num_layers):
            ## Item tower
            if i == 0:
                item_rep = itemEmbeddings
                next_item_rep, _ = self.self_attention[i](item_rep, item_rep, item_rep, non_pad_mask=non_pad_mask, attn_mask=mask)
            else:
                item_rep, _ = self.item_gita[i](item_rep, next_item_rep, next_item_rep, time_rep, next_time_rep, non_pad_mask=non_pad_mask, attn_mask=mask)
                next_item_rep = self.item_func(item_rep, time_rep, next_time_rep, self.item_mlp1[i], self.item_mlp2[i], self.item_mlp3[i], self.item_ln1[i], self.item_ln2[i], self.item_ln3[i])

            ## Time tower
            if i == 0:
                time_rep, _ = self.time_gita[i](fourierEncoding, fourierEncoding, fourierEncoding, item_rep, next_item_rep, non_pad_mask=non_pad_mask, attn_mask=mask)
                next_time_rep = self.time_func(time_rep, item_rep, next_item_rep, self.time_mlp1[i], self.time_mlp2[i], self.time_mlp3[i], self.time_ln1[i], self.time_ln2[i], self.time_ln3[i])
            else:
                time_rep, _ = self.time_gita[i](time_rep, next_time_rep, next_time_rep, item_rep, next_item_rep, non_pad_mask=non_pad_mask, attn_mask=mask)
                next_time_rep = self.time_func(time_rep, item_rep, next_item_rep, self.time_mlp1[i], self.time_mlp2[i], self.time_mlp3[i], self.time_ln1[i], self.time_ln2[i], self.time_ln3[i])

            next_item_rep_list.append(next_item_rep)
            next_time_rep_list.append(next_time_rep)

        next_time_rep_list = torch.stack(next_time_rep_list, dim=0) 
        next_item_rep_list = torch.stack(next_item_rep_list, dim=0) 

        final_next_time_rep = next_time_rep_list[-1] * isValid.unsqueeze(2) 
        final_next_item_rep = next_item_rep_list[-1] * isValid.unsqueeze(2)


        # next item prediction
        allItemEmbeddings = self.item_embedding.weight 
        item_output = torch.matmul(final_next_item_rep, allItemEmbeddings.t())  

        # time loss & contrastive loss
        if not test:
            final_next_time_rep_ = self.final_time_layer(final_next_time_rep) 
            final_next_time_rep_ = final_next_time_rep_.reshape(batch_size, max_length, -1, 2)
            final_next_time_rep_ = F.normalize(final_next_time_rep_, dim=-1) 

            fourierEncoding_pos = self.calendar_fourier_encoding(month_labels, wday_labels, hour_labels, isValid)

            # time loss
            fourierEncoding_pos_ = fourierEncoding_pos.reshape(batch_size, max_length, -1, 2)
            kappa = F.softplus(self.log_concentration) + 1e-4
            log_i0 = torch.log(torch.special.i0e(kappa) + 1e-12) + kappa
            time_loss = torch.mean((-kappa * torch.sum(final_next_time_rep_ * fourierEncoding_pos_, dim=-1) + log_i0) * isValid.unsqueeze(2))

            # contrastive loss
            month_negs = torch.randint(low=1, high=12+1, size=(batch_size, max_length), dtype=torch.long, device=self.device) 
            wday_negs = torch.randint(low=1, high=7+1, size=(batch_size, max_length), dtype=torch.long, device=self.device) 
            hour_negs = torch.randint(low=1, high=24+1, size=(batch_size, max_length), dtype=torch.long, device=self.device) 
            fourierEncoding_neg = self.calendar_fourier_encoding(month_negs, wday_negs, hour_negs, isValid)
            fourierEncoding_pos_ = F.normalize(self.fourier_mlp2(torch.relu(self.fourier_mlp1(fourierEncoding_pos))), dim=-1) 
            fourierEncoding_neg_ = F.normalize(self.fourier_mlp2(torch.relu(self.fourier_mlp1(fourierEncoding_neg))), dim=-1) 
            contrastive_score_pos = torch.sum(F.normalize(final_next_item_rep, dim=-1) * fourierEncoding_pos_.detach(), dim=-1) 
            contrastive_score_neg = torch.sum(F.normalize(final_next_item_rep, dim=-1) * fourierEncoding_neg_.detach(), dim=-1) 
            closs = -F.logsigmoid(contrastive_score_pos - contrastive_score_neg) * isValid
            closs = closs.mean() 

        else: 
            time_loss = None
            closs = None

        return item_output, closs, time_loss
