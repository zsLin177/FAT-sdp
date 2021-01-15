import os

import torch
import torch.nn as nn
from supar.models import BiaffineSemanticDependencyModel
from supar.modules import LSTM
from supar.parsers.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import bos, pad, unk
from supar.utils.field import ChartField, Field, SubwordField, TransitionField, StateField, ParStateField, ParTransitionField, TranLabelField
from supar.utils.logging import get_logger, progress_bar
from supar.utils.metric import ChartMetric
from supar.utils.transform import CoNLL
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random

# x = torch.rand(2, 3, 4)
# m = nn.Dropout(p=0.2)
# print(x)
# out = m(x)
# print(out)

# x = torch.rand(2, 7, requires_grad=True)
# y = torch.tensor([3, 4], dtype=torch.long)
# crition = nn.CrossEntropyLoss()
# loss = crition(x, y)
# print(loss)
# loss1 = crition(x[0].unsqueeze(0), torch.tensor([3], dtype=torch.long))
# loss2 = crition(x[1].unsqueeze(0), torch.tensor([4], dtype=torch.long))
# lst = [loss1, loss2]
# loss_plus = torch.sum(torch.stack(lst))
# print(loss_plus)

# x = torch.rand(2, 4, 3, 10)
# inter = torch.tensor([1, 0, 1])
# inter1 = inter.reshape(3, 1)
# inter1 = inter1.expand(-1, 10)
# print(inter.shape)

# x = torch.tensor([1, 2])
# print(x.device)

# with torch.no_grad():
#     h_t = torch.zeros(1, 2, 4)
#     c_t = torch.zeros(1, 2, 4)
#     input_x = torch.rand(2, 3, 3)
#     x_len = torch.tensor([2, 1])
#     lstm = nn.LSTM(input_size=3, hidden_size=4, bidirectional=False)
#     x = pack_padded_sequence(input_x, x_len, True, False)
#     x, _ = lstm(x, (h_t, c_t))
#     x, _ = pad_packed_sequence(x, True, total_length=3)
#     print(x)
#     print(x.shape)

#     h_t = torch.zeros(1, 1, 4)
#     c_t = torch.zeros(1, 1, 4)
#     b1 = input_x[0].detach()
#     for i in range(2):
#         y = b1[i].unsqueeze(0).unsqueeze(0)
#         out, (h_t, c_t) = lstm(y, (h_t, c_t))
#         print(out.squeeze(0).squeeze(0))

# states = torch.ones(2, 8, 4, 4) * (-1)
# windowed = states[..., :2]
# win_states = windowed[:, :, 0:3, :]
# print(win_states)
# null_lstm_mask = win_states.eq(-1)
# # win_states = win_states.masked_fill(null_lstm_mask, 10)
# win_states.masked_fill_(null_lstm_mask, 10)
# print(win_states)
# print(windowed)

# transitions_repr = torch.randn(10, 2)
# x = transitions_repr.transpose(1, 0)
# print(transitions_repr)
# print(x)
# activation = nn.functional.relu
# x = activation(transitions_repr)
# print(x)
# print(transitions_repr)
# x = transitions_repr.transpose(1, 0)
# print(transitions_repr.shape)
# print(x.shape)

# states_repr = torch.rand(3, 8, 20)
# out = torch.matmul(states_repr, transitions_repr)
# print(out.shape)

# hidden = torch.rand(9, 10, 10)
# hidden = hidden.reshape(9, -1)
# print(hidden.shape)

# x = torch.rand(5)
# idx = torch.argmax(x, dim=0)
# print(x)
# print(idx.item())

# hidden = torch.rand(9, 10)
# x = [0]
# y = hidden[x]
# print(y)
# y = torch.sum(y, dim=0)
# print(y)

# x = [1, 2, 3, 4]
# y = x[-2:]
# print(y)

# hidden = torch.rand(3, 9, 10)
# print(hidden[0])
# print(hidden[0].shape)

# words = torch.tensor([[1, 2, 3, 0, 0], [3, 4, 5, 6, 0]])
# mask = words.ne(0)
# seq_len = torch.sum(mask, dim=-1)
# print(seq_len.shape)
# print(seq_len[0].item())

# x = torch.randint(0, 10, (2, 6, 10))
# idx = torch.randint(0, 6, (2, 3, 2))
# print(x)
# print(idx)
# idx = idx.view(2, 3*2, 1).expand(-1, -1, 10)
# print(idx)

# x = torch.rand(2, 4, 10)
# y = torch.tensor([[0, 1, 2, -1], [4, 2, -1, -1]])
# print(x)
# print(y)
# mask = y.ge(0)
# print(mask)
# x_ = x[mask]
# y_ = y[mask]
# print(x_)
# print(y_)

# embed = nn.Embedding(num_embeddings=3, embedding_dim=10)
# x = embed(torch.tensor([0, 1, 2]))
# print(x)
# print(x.shape)

# x = torch.ones(1, 2, 3, 10)
# x = torch.sum(x, dim=2)
# print(x)

# x = torch.ones(2, 3, 10)
# y = torch.ones(1, 1, 10)
# y_ = y.expand(2, -1, -1)
# z = torch.cat([x, y_], dim=1)
# print(z)

# x = torch.rand(1, 7, 10)
# print(x)
# idx = torch.randint(0, 7, (1, 2, 3*2))
# # print(idx)
# idx = idx.resize(1, 2*6).unsqueeze(-1)
# # print(idx)
# idx = idx.expand(-1, -1, 10)
# print(idx)

# y = torch.gather(x, 1, idx)
# print(y)

# x = torch.tensor([[1, 2, 3, -1], [0, 2, -1, -1]])
# mask = x.eq(-1)
# print(mask)
# x.masked_fill_(mask, 4)
# print(x)

# x = torch.zeros(1, 1, 10)
# print(x)

# batch_size = 3
# seq_len = 6

# null_tensor = torch.rand(1, 1, 10)
# print(null_tensor)
# null_tensor = null_tensor.expand(3, -1, -1)
# print(null_tensor)
# x = torch.rand(3, 6, 100)
# y = torch.cat([x, null_tensor], dim=1)
# print(y.shape)

# batch_size = 3
# paded_t_len = 6
# just_len = 5
# x = torch.range(1, 360)
# x.resize_(3, 6, 4, 5)
# print(x)

# y = x[..., :2]
# print(y)

# y = x[:, :, 0:3, :]
# z = x[:, :, 3:, :].squeeze(2)
# print(y)
# print(z)
# print(z.shape)

# train_path = 'data/sdp/DM/dev.conllu'
# WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
# TAG = Field('tags', bos=bos)
# CHAR = SubwordField('chars', pad=pad, unk=unk, bos=bos, fix_len=20)
# LEMMA = Field('lemmas', pad=pad, unk=unk, bos=bos, lower=True)
# EDGE = ChartField('edges', use_vocab=False, fn=CoNLL.get_edges)
# LABEL = ChartField('labels', fn=CoNLL.get_labels)
# Transition = TransitionField('transitions', label_field=LABEL)
# State = StateField('states', label_field=LABEL, transition_field=Transition)
# transform = CoNLL(FORM=(WORD, CHAR, None),
#                   LEMMA=LEMMA,
#                   POS=TAG,
#                   PHEAD=(EDGE, LABEL, Transition, State))

# train = Dataset(transform, train_path)
# WORD.build(train, 7, (None))
# TAG.build(train)
# CHAR.build(train)
# LEMMA.build(train)
# LABEL.build(train)
# Transition.build(train)

# train.build(3000, 32)
# batch = next(iter(train.loader))
# print(batch)
# for x in batch:
#     print(x.shape)

# print(Transition.vocab.stoi)


# def padding(tensors, padding_value=0, total_length=None):
#     size = [len(tensors)] + [
#         max(tensor.size(i) for tensor in tensors)
#         for i in range(len(tensors[0].size()))
#     ]
#     if total_length is not None:
#         assert total_length >= size[1]
#         size[1] = total_length
#     out_tensor = tensors[0].data.new(*size).fill_(padding_value)
#     for i, tensor in enumerate(tensors):
#         out_tensor[i][[slice(0, i) for i in tensor.size()]] = tensor
#     return out_tensor


# seq_len = 10
# batch_size = 3
# words_len = torch.tensor([5, 7, 10], dtype=torch.long)
# x = torch.tensor([[0] + [-1] * (seq_len - 1), [-1] * seq_len, [-1] * seq_len],
#                  dtype=torch.long)
# x = x.unsqueeze(0).expand(batch_size, -1, -1)

# y = []
# for length in words_len:
#     y.append(torch.arange(1, length.item(), dtype=torch.long))

# out = padding(y, -1, 10).reshape(batch_size, 1, -1)
# init_stat = torch.cat((x, out), dim=1)
# init_stat = init_stat[:, [0, 3, 2, 1]]
# print(init_stat)

# init_stat[1, 1, 0] = -1
# init_stat[0, 1, 0] = -1
# init_stat[2, 1, 0] = -1

# mask = init_stat[:, 1, 0].gt(-1)
# print(mask)
# print(not mask.sum())
# remain = init_stat[mask]
# print(remain)

# x = torch.ones((3, 7), dtype=torch.long)
# print(x)
# mask = torch.tensor([True, False, False]).reshape(3, -1)
# print(mask)
# m1 = x.gt(0) & mask
# print(m1)

# col_cond1 = torch.zeros(7).index_fill_(-1, torch.tensor([2, 4, 6]), 1).gt(0)
# m2 = m1 * col_cond1
# x.masked_fill_(~m2, 0)
# print(x)

# x = torch.randint(-1, 3, (2, 4, 10))
# print(x)
# mask = x[:, 0, 0] == x[:, 1, 0]
# print(mask)

# x = torch.randint(0, 10, (4, 4, 7))
# edge = torch.randint(-1, 1, (4, 10, 10))

# stack_isnot_clear = torch.tensor([True, False, False, True]).int()
# print(stack_isnot_clear)
# idx = (stack_isnot_clear == 1).nonzero().squeeze(1)
# print(idx)

# mask = torch.tensor([True, False, False, True])
# print(x[mask])

# seq_len = 10
# x = torch.randint(-1, 10, (4, 4, 10))
# edge = torch.randint(0, 2, (4, 10, 10))
# stack_head = x[:, 0, 0]
# print(stack_head)
# condicate_b = x[:, 1, 1:]
# print(condicate_b)
# stack_head = stack_head.unsqueeze(1).expand(-1, seq_len-1)
# print(stack_head)
# print(edge)

# import json
# lst1 = [1, 2, 3]
# lst2 = [3, 2, 1, 0]
# lst = [lst1, lst2]
# with open ('happy.json', 'w') as f:
#     json.dump(lst, f)
    




