#!/usr/bin/env python37
# -*- coding: utf-8 -*-

import argparse
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import metric
from model import *
from tqdm import *
from preprocess import preprocess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def str2bool(v): 
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: 
        return True

#################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Movielens1M')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--embed_dim', type=int, default=128)
parser.add_argument('--month', default=True, type=str2bool)
parser.add_argument('--wday', default=True, type=str2bool)
parser.add_argument('--hour', default=True, type=str2bool)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--max_length', type=int, default=50)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--num_epoch', type=int, default=500)
args = parser.parse_args()

batch_size = args.batch_size
embed_dim = args.embed_dim
lr = args.lr
month = args.month
wday = args.wday
hour = args.hour
dropout = args.dropout
patience = args.patience
max_length = args.max_length
num_layers = args.num_layers
num_epoch = args.num_epoch

k = [5, 10, 20]

##################################################################################

print("batch_size     : " + str(batch_size))
print("embed_dim      : " + str(embed_dim))
print("lr             : " + str(lr))
print("month          : " + str(month))
print("wday           : " + str(wday))
print("hour           : " + str(hour))
print("dropout        : " + str(dropout))
print("patience       : " + str(patience))
print("max_length     : " + str(max_length))
print("num_layers     : " + str(num_layers))
print("num_epoch      : " + str(num_epoch))

def print_result(perf):
    print('Recall@5 : ' + str(round(perf[0], 4)) + ', NDCG@5 : ' + str(round(perf[1], 4)))
    print('Recall@10: ' + str(round(perf[2], 4)) + ', NDCG@10: ' + str(round(perf[3], 4)))
    print('Recall@20: ' + str(round(perf[4], 4)) + ', NDCG@20: ' + str(round(perf[5], 4))) 
    print('')

def main():
    print('Loading data...')
    file_name = f"./{args.dataset}/data.txt"
    
    user_sequence, item_sequences, month_sequences, wday_sequences, hour_sequences, timestamp_sequences, \
        num_users, num_items = preprocess(args.dataset, file_name, max_length)
    
    print(f"# users: {num_users}")
    print(f"# items: {num_items}")

    model = DTTRec(num_items, embed_dim, max_length-3, num_layers, dropout, month=month, wday=wday, hour=hour).to(device)
    optimizer = optim.AdamW(model.parameters(), lr, weight_decay=0.0) 

    best_perf = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_recall_i = 0
    
    for epoch in range(1, num_epoch+1):
        # train for one epoch
        train_loss = trainForEpoch(item_sequences, month_sequences, wday_sequences, hour_sequences, model, optimizer, batch_size, epoch)

        val_perf = validate(item_sequences, month_sequences, wday_sequences, hour_sequences, model, k, epoch, validation=True)

        print("Validation")
        print_result(val_perf)
        
        if val_perf[3] > best_perf[3]: # ndcg@10
            best_perf = list(val_perf)
            best_recall_i = epoch
            test_perf = validate(item_sequences, month_sequences, wday_sequences, hour_sequences, model, k, epoch, validation=False)
            print("Test")
            print_result(test_perf)

        print("Best_validation (" + str(best_recall_i) + ")")
        print_result(best_perf)

        if best_recall_i + patience < epoch:
            break

    print("Best_validation (" + str(best_recall_i) + ")")
    print_result(best_perf)

    print("Best Test")
    print_result(test_perf)


def trainForEpoch(item_sequences, month_sequences, wday_sequences, hour_sequences, model, optimizer, batch_size, epoch):
    model.train()
    data = list(zip(item_sequences, month_sequences, wday_sequences, hour_sequences))
    random.shuffle(data)
    item_sequences, month_sequences, wday_sequences, hour_sequences = zip(*data)

    loss_func = nn.CrossEntropyLoss(ignore_index=0, reduction='none').to(device)
    total_item_loss = 0.0
    total_closs = 0.0
    total_time_loss = 0.0
    num_batches = 0

    with trange(0, len(item_sequences), batch_size) as t:
        for i in t:
            t.set_description("Training Epoch %d Batch %d" % (epoch, i))
            optimizer.zero_grad()
            
            batch_sequences = item_sequences[i: i + batch_size]
            batch_months = month_sequences[i: i + batch_size]
            batch_wdays = wday_sequences[i: i + batch_size]
            batch_hours = hour_sequences[i: i + batch_size]

            tensor_sequences = torch.LongTensor(batch_sequences)[:, :-3].to(device) # batch_size * max_len
            tensor_months = torch.LongTensor(batch_months)[:, :-3].to(device) # batch_size * max_len
            tensor_wdays = torch.LongTensor(batch_wdays)[:, :-3].to(device) # batch_size * max_len
            tensor_hours = torch.LongTensor(batch_hours)[:, :-3].to(device) # batch_size * max_len

            item_labels = torch.LongTensor(batch_sequences)[:, 1:-2].to(device) # batch_size * max_len
            hour_labels = torch.LongTensor(batch_hours)[:, 1:-2].to(device) # batch_size * max_len
            wday_labels = torch.LongTensor(batch_wdays)[:, 1:-2].to(device) # batch_size * max_len
            month_labels = torch.LongTensor(batch_months)[:, 1:-2].to(device) # batch_size * max_len

            batch_size_ = tensor_sequences.shape[0]
            num_batches += 1

            # model
            item_output, closs, time_loss = model(tensor_sequences, tensor_months, tensor_wdays, tensor_hours, month_labels, wday_labels, hour_labels) 

            item_loss = loss_func(item_output.reshape(item_output.shape[0] * item_output.shape[1], -1), item_labels.reshape(-1))
            loss = item_loss + 0.0001 * time_loss + 0.001 * closs 
            loss = loss.mean()

            loss.backward()
            optimizer.step() 
            total_item_loss += item_loss.sum().item() / torch.count_nonzero(item_labels).item()

            t.set_postfix(item=total_item_loss / num_batches)

    return total_item_loss


def validate(item_sequences, month_sequences, wday_sequences, hour_sequences, model, k, epoch, validation):
    model.eval()

    item_perf = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    num_instances = 0
    batch_size = 128

    with torch.no_grad():
        with trange(0, len(item_sequences), batch_size) as t:
            for i in t:
                if validation:
                    t.set_description("Validation Epoch %d Batch %d" % (epoch, i))
                else:
                    t.set_description("Test Epoch %d Batch %d" % (epoch, i))

                batch_sequences = item_sequences[i: i + batch_size]
                batch_months = month_sequences[i: i + batch_size]
                batch_wdays = wday_sequences[i: i + batch_size]
                batch_hours = hour_sequences[i: i + batch_size]

                if validation: ## Validation
                    tensor_sequences = torch.LongTensor(batch_sequences)[:, 1:-2].to(device) # batch_size * max_len
                    tensor_months = torch.LongTensor(batch_months)[:, 1:-2].to(device) # batch_size * max_len
                    tensor_wdays = torch.LongTensor(batch_wdays)[:, 1:-2].to(device) # batch_size * max_len
                    tensor_hours = torch.LongTensor(batch_hours)[:, 1:-2].to(device) # batch_size * max_len
                    item_labels = torch.LongTensor(batch_sequences)[:, -2].to(device) # batch_size
                    hour_labels = torch.LongTensor(batch_hours)[:, -2].to(device) # batch_size
                    wday_labels = torch.LongTensor(batch_wdays)[:, -2].to(device) # batch_size
                    month_labels = torch.LongTensor(batch_months)[:, -2].to(device) # batch_size
                else: ## Test
                    tensor_sequences = torch.LongTensor(batch_sequences)[:, 2:-1].to(device) # batch_size * max_len
                    tensor_months = torch.LongTensor(batch_months)[:, 2:-1].to(device) # batch_size * max_len
                    tensor_wdays = torch.LongTensor(batch_wdays)[:, 2:-1].to(device) # batch_size * max_len
                    tensor_hours = torch.LongTensor(batch_hours)[:, 2:-1].to(device) # batch_size * max_len
                    item_labels = torch.LongTensor(batch_sequences)[:, -1].to(device) # batch_size
                    hour_labels = torch.LongTensor(batch_hours)[:, -1].to(device) # batch_size
                    wday_labels = torch.LongTensor(batch_wdays)[:, -1].to(device) # batch_size
                    month_labels = torch.LongTensor(batch_months)[:, -1].to(device) # batch_size

                num_instances += tensor_sequences.shape[0]
                
                # model
                item_output, _, _ = model(tensor_sequences, tensor_months, tensor_wdays, tensor_hours, month_labels, wday_labels, hour_labels, test=True) # batch_size * num_items
                item_output = item_output[:, -1, :] # b * I+1

                item_logits = F.softmax(item_output, dim = 1)
                item_recall, item_ndcg = metric.evaluate(item_logits, item_labels, k=k)

                item_perf[0] += item_recall[0]
                item_perf[1] += item_ndcg[0]
                item_perf[2] += item_recall[1]
                item_perf[3] += item_ndcg[1]
                item_perf[4] += item_recall[2]
                item_perf[5] += item_ndcg[2]

                t.set_postfix(item_n10=item_perf[3] / num_instances)                

    item_perf = list(map(lambda x: x / num_instances, item_perf))

    return item_perf


if __name__ == '__main__':
    main()
