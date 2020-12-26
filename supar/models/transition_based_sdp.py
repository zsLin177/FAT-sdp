# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.modules import LSTM, MLP, BertEmbedding, Biaffine, CharLSTM
from supar.modules.dropout import IndependentDropout, SharedDropout
from supar.utils import Config
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from supar.utils.fn import pad
import pdb


class TransitionSemanticDependencyModel(nn.Module):
    r"""
    The implementation of Transition-based Semantic Dependency Parser.

    References:
        -Yuxuan Wang, Wanxiang Che, etc 2018.
          `A Neural Transition-Based Approach for Semantic Dependency Graph Parsing`_.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_labels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, needed if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, needed if character-level representations are used. Default: ``None``.
        n_lemmas (int):
            The number of lemmas, needed if lemma embeddings are used. Default: ``None``.
        feat (str):
            Additional features to use，separated by commas.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'lemma'``: Lemma embeddings.
            ``'bert'``: BERT representations, other pretrained langugae models like XLNet are also feasible.
            Default: ``'tag,char,lemma'``.
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_embed_proj (int):
            The size of linearly transformed word embeddings. Default: 125.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if ``feat='char'``. Default: 50.
        char_pad_index (int):
            The index of the padding token in the character vocabulary. Default: 0.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'`` and ``'xlnet-base-cased'``.
            This is required if ``feat='bert'``. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use. Required if ``feat='bert'``.
            The final outputs would be the weight sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers. Required if ``feat='bert'``. Default: .0.
        bert_pad_index (int):
            The index of the padding token in the BERT vocabulary. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .2.
        n_lstm_hidden (int):
            The size of LSTM hidden states. Default: 600.
        n_lstm_layers (int):
            The number of LSTM layers. Default: 3.
        lstm_dropout (float):
            The dropout ratio of LSTM. Default: .33.
        n_mlp_edge (int):
            Edge MLP size. Default: 600.
        n_mlp_label  (int):
            Label MLP size. Default: 600.
        edge_mlp_dropout (float):
            The dropout ratio of edge MLP layers. Default: .25.
        label_mlp_dropout (float):
            The dropout ratio of label MLP layers. Default: .33.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """
    def __init__(self,
                 n_words,
                 n_labels,
                 n_transitions,
                 transition_vocab,
                 decode_mode,
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 feat='tag,char,lemma',
                 n_embed=100,
                 n_transition_embed=600,
                 n_feat_embed=100,
                 n_char_embed=50,
                 char_pad_index=0,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pad_index=0,
                 embed_dropout=.2,
                 n_lstm_hidden=600,
                 n_lstm_layers=3,
                 lstm_dropout=.33,
                 n_mlp=600,
                 mlp_dropout=.25,
                 pad_index=0,
                 unk_index=1,
                 window=1,
                 **kwargs):
        super().__init__()

        self.args = Config().update(locals())
        self.state_pad_idx = -1
        self.n_transitions = n_transitions
        self.window = window
        self.decode_mode = decode_mode
        self.transition_vocab = transition_vocab
        self.n_labels = n_labels  # 不算空label的数量
        # the embedding layer
        self.word_embed = nn.Embedding(num_embeddings=n_words,
                                       embedding_dim=n_embed)
        self.n_input = n_embed
        if 'tag' in feat:
            self.tag_embed = nn.Embedding(num_embeddings=n_tags,
                                          embedding_dim=n_feat_embed)
            self.n_input += n_feat_embed
        if 'char' in feat:
            self.char_embed = CharLSTM(n_chars=n_chars,
                                       n_embed=n_char_embed,
                                       n_out=n_feat_embed,
                                       pad_index=char_pad_index)
            self.n_input += n_feat_embed
        if 'lemma' in feat:
            self.lemma_embed = nn.Embedding(num_embeddings=n_lemmas,
                                            embedding_dim=n_feat_embed)
            self.n_input += n_feat_embed
        if 'bert' in feat:
            self.bert_embed = BertEmbedding(model=bert,
                                            n_layers=n_bert_layers,
                                            pad_index=bert_pad_index,
                                            dropout=mix_dropout)
            self.n_input += self.bert_embed.n_out
        self.embed_dropout = IndependentDropout(p=embed_dropout)

        # the transition embedding layer
        self.transition_embed = nn.Embedding(num_embeddings=n_transitions + 1,
                                             embedding_dim=n_transition_embed)
        self.transition_embed_droupout = IndependentDropout(p=embed_dropout)
        self.new_droupout = nn.Dropout(p=embed_dropout)

        # the lstm layer
        self.lstm = LSTM(input_size=self.n_input,
                         hidden_size=n_lstm_hidden,
                         num_layers=n_lstm_layers,
                         bidirectional=True,
                         dropout=lstm_dropout)
        self.lstm_dropout = SharedDropout(p=lstm_dropout)
        # null_lstm_hidden
        self.null_lstm_hidden = nn.Parameter(
            torch.zeros(1, 1, n_lstm_hidden * 2))

        if (decode_mode == 'lstm'):
            # decoder lstm
            self.decoder_lstm = nn.LSTM(
                input_size=n_lstm_hidden * 2 * 3 + n_transition_embed,
                hidden_size=(n_lstm_hidden * 2 * 3 + n_transition_embed) // 3,
                num_layers=1,
                bidirectional=False)
            self.decoder_lstm_dropout = SharedDropout(p=lstm_dropout)
            self.decoder_lstm_mlp = MLP(
                n_in=(n_lstm_hidden * 2 * 3 + n_transition_embed) // 3,
                n_out=n_transitions,
                dropout=mlp_dropout,
                activation=False)

        elif (decode_mode == 'mlp'):
            # the MLP layer
            self.base_mlp = MLP(n_in=n_lstm_hidden * 2 * 3 +
                                n_transition_embed,
                                n_out=n_transitions,
                                dropout=mlp_dropout,
                                activation=False)

        elif (decode_mode == 'att'):
            self.state_mlp = MLP(n_in=n_lstm_hidden * 2 * 3 +
                                 n_transition_embed,
                                 n_out=n_mlp,
                                 dropout=mlp_dropout,
                                 activation=False)
            self.action_mlp = MLP(n_in=n_transition_embed,
                                  n_out=n_mlp,
                                  dropout=mlp_dropout,
                                  activation=False)

        elif (decode_mode == 'beta'):
            self.inter = nn.Parameter(torch.tensor([1., 1., 1.]))
            self.mlp = MLP(n_in=n_lstm_hidden * 2 + n_transition_embed,
                           n_out=n_transitions,
                           dropout=mlp_dropout,
                           activation=False)

        elif (decode_mode == 'dual'):
            self.action_mlp = MLP(n_in=n_lstm_hidden * 2 * 3 +
                                  n_transition_embed,
                                  n_out=n_transitions,
                                  dropout=mlp_dropout,
                                  activation=False)
            self.label_mlp = MLP(n_in=n_lstm_hidden * 2 * 3 +
                                 n_transition_embed * 2,
                                 n_out=n_labels + 1,
                                 dropout=mlp_dropout,
                                 activation=False)

        self.activation = nn.functional.relu

        self.criterion = nn.CrossEntropyLoss()
        self.pad_index = pad_index
        self.unk_index = unk_index

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)
        return self

    def dynamic_loss(self, words, words_len, feats, edges, labels):
        r"""
        目前的explore处理的粒度是整个batch，之后应该会试试step粒度的explore
        目前只适用dual
        目前的优化目标只是Correct(c)中分数最高的
        words: [batch_size, seq_len]
        words_len: [batch_size]
        edges : [batch_size, seq_len, seq_len]
        labels: [batch_size, seq_len, seq_len]
        """

        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)

        feat_embeds = []
        if 'tag' in self.args.feat:
            feat_embeds.append(self.tag_embed(feats.pop()))
        if 'char' in self.args.feat:
            feat_embeds.append(self.char_embed(feats.pop(0)))
        if 'bert' in self.args.feat:
            feat_embeds.append(self.bert_embed(feats.pop(0)))
        if 'lemma' in self.args.feat:
            feat_embeds.append(self.lemma_embed(feats.pop(0)))
        word_embed, feat_embed = self.embed_dropout(word_embed,
                                                    torch.cat(feat_embeds, -1))
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        x = pack_padded_sequence(embed, mask.sum(1), True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)
        # x:[batch_size, seq_len, hidden_size*2]
        null_hidden = self.null_lstm_hidden.expand(batch_size, -1, -1)
        x = torch.cat([x, null_hidden], dim=1)
        # x:[batch_size, seq_len+1, hidden_size*2]
        lstm_out_size = x.shape[2]

        action_id = {
            action_: [
                self.transition_vocab.stoi[a]
                for a in self.transition_vocab.stoi.keys()
                if a.startswith(action_)
            ]
            for action_ in ["LR", "LP", "RS", "RP", "NS", "NR", "NP"]
        }

        losses = []
        for i in range(batch_size):
            look_up = x[i]
            edge = edges[i].cpu().numpy()
            label = labels[i].cpu().numpy()
            loss = []

            # arc = [[0] * seq_len for k in range(seq_len)]
            # label = [[-1] * seq_len for k in range(seq_len)]

            actions = [self.n_transitions] * self.window  # 动作序列,-1头
            stack = [seq_len] * self.window + [0]  # 正在处理的节点序列,-1头
            buffer = [k for k in range(1, words_len[i].item())
                      ] + [seq_len] * self.window  # 待处理的节点序列,0头
            deque = [seq_len] * self.window  # 暂时跳过的节点序列（可能存在多个头节点）,-1头
            buffer_len = words_len[i].item() - 1
            stack_len = 1
            count = 0

            while (buffer_len > 0):
                if (count > 500):
                    print('count supass 500')
                    break
                count += 1

                valid_actions = []
                if (stack_len > 1 and buffer_len > 0):
                    valid_actions += action_id['LR']
                    valid_actions += action_id['LP']
                    valid_actions += action_id['RP']

                if (buffer_len > 0):
                    valid_actions += action_id['NS']
                    if (stack_len > 0):
                        valid_actions += action_id['RS']  # ROOT,NULL

                if (stack_len > 1):
                    valid_actions += action_id['NR']
                    valid_actions += action_id['NP']

                correct_actions = []  #用来存储在当前情况下，cost为0的transition
                correct_labels = []  #用来存储在当前情况下，正确transition对应的label序列
                stack_head = stack[-1]
                buffer_head = buffer[0]
                # 这个地方可能要判断stack_head是否是seq_len
                if (stack_head == seq_len):
                    correct_actions += action_id['NS']
                    correct_labels.append([self.n_labels])
                else:
                    flag = 1
                    if (edge[buffer_head][stack_head]):
                        flag = 0
                    else:
                        for b in buffer[1:-self.window]:
                            if (edge[stack_head][b] or edge[b][stack_head]):
                                flag = 0
                                break
                    if (flag):
                        # LR这个action是可以的,但是还没考虑label
                        correct_actions += action_id['LR']
                        if (label[stack_head][buffer_head] != -1):
                            correct_labels.append(
                                [label[stack_head][buffer_head]])
                        else:
                            # correct_labels.append(
                            #     [temp for temp in range(self.n_labels)])
                            # 我突然觉得这个不应该是所有的，应该是随机一个或者就是[self.n_labels]
                            correct_labels.append([self.n_labels])

                    flag = 1
                    if (edge[stack_head][buffer_head]):
                        flag = 0
                    else:
                        for s in stack[self.window:-1]:
                            if (edge[s][buffer_head] or edge[buffer_head][s]):
                                flag = 0
                                break
                    if (flag):
                        # RS
                        correct_actions += action_id['RS']
                        if (label[buffer_head][stack_head] != -1):
                            correct_labels.append(
                                [label[buffer_head][stack_head]])
                        else:
                            correct_labels.append([self.n_labels])

                    flag = 1
                    for a in stack[self.window:]:
                        if (edge[a][buffer_head] or edge[buffer_head][a]):
                            flag = 0
                            break
                    if (flag):
                        # NS
                        correct_actions += action_id['NS']
                        correct_labels.append([self.n_labels])  #加入空标签对应的下标

                    flag = 1
                    for b in buffer[:-self.window]:
                        if (edge[b][stack_head] or edge[stack_head][b]):
                            flag = 0
                            break
                    if (flag):
                        # NR
                        correct_actions += action_id['NR']
                        correct_labels.append([self.n_labels])

                    flag = 1
                    if (edge[buffer_head][stack_head]):
                        flag = 0
                    if (flag):
                        # LP
                        correct_actions += action_id['LP']
                        if (label[stack_head][buffer_head] != -1):
                            correct_labels.append(
                                [label[stack_head][buffer_head]])
                        else:
                            correct_labels.append([self.n_labels])

                    flag = 1
                    if (edge[stack_head][buffer_head]):
                        flag = 0
                    if (flag):
                        # RP
                        correct_actions += action_id['RP']
                        if (label[buffer_head][stack_head] != -1):
                            correct_labels.append(
                                [label[buffer_head][stack_head]])
                        else:
                            correct_labels.append([self.n_labels])

                    flag = 1
                    if (edge[buffer_head][stack_head]
                            or edge[stack_head][buffer_head]):
                        flag = 0
                    if (flag):
                        # NP
                        correct_actions += action_id['NP']
                        correct_labels.append([self.n_labels])

                correct_idx = [
                    idx for idx in range(len(correct_actions))
                    if correct_actions[idx] in valid_actions
                ]
                # print('idxs', correct_idx)
                # print('actions', correct_actions)
                # print('labels', correct_labels)
                correct_actions_ = [
                    correct_actions[idx] for idx in correct_idx
                ]
                correct_labels_ = [correct_labels[idx] for idx in correct_idx]
                # 怀疑这个correct_actions可能为空

                action = valid_actions[0]
                o_action = correct_actions_[0]
                o_labels = correct_labels_[0]

                stack_embed = torch.sum(look_up[stack[-self.window:]], dim=0)
                buffer_embed = torch.sum(look_up[buffer[0:self.window]], dim=0)
                deque_embed = torch.sum(look_up[deque[-self.window:]], dim=0)
                last_t_embed = self.transition_embed(
                    torch.tensor(actions[-self.window:],
                                 dtype=torch.long,
                                 device=x.device))
                # last_t_embed_droup = self.transition_embed_droupout(last_t_embed)[0]
                last_t_embed_droup = self.new_droupout(last_t_embed)
                transition_embed = torch.sum(last_t_embed_droup, dim=0)
                # pdb.set_trace()
                final_repr = torch.cat(
                    (transition_embed, stack_embed, buffer_embed, deque_embed),
                    dim=-1).unsqueeze(0)

                score = self.action_mlp(final_repr)

                if (len(valid_actions) > 1):
                    action_idx = torch.argmax(score.squeeze(0)[torch.tensor(
                        valid_actions, dtype=torch.long, device=x.device)],
                                              dim=0).item()
                    action = valid_actions[action_idx]

                if (len(correct_actions_) > 1):
                    o_action_idx = torch.argmax(score.squeeze(0)[torch.tensor(
                        correct_actions_, dtype=torch.long, device=x.device)],
                                                dim=0).item()
                    o_action = correct_actions_[o_action_idx]
                    o_labels = correct_labels_[o_action_idx]

                action_loss = self.criterion(
                    score,
                    torch.tensor([o_action],
                                 dtype=torch.long,
                                 device=score.device))

                current_action_embed = self.transition_embed(
                    torch.tensor([action], dtype=torch.long, device=x.device))
                label_final = torch.cat((final_repr, current_action_embed), -1)
                label_score = self.label_mlp(label_final)

                if (len(o_labels) > 1):
                    label_idx = torch.argmax(
                        label_score.squeeze(0)[torch.tensor(
                            o_labels,
                            dtype=torch.long,
                            device=label_score.device)],
                        dim=0).item()
                    y_label = o_labels[label_idx]
                else:
                    y_label = o_labels[0]

                label_loss = self.criterion(
                    label_score,
                    torch.tensor([y_label],
                                 dtype=torch.long,
                                 device=label_score.device))

                loss_step = label_loss + action_loss
                loss.append(loss_step)

                actions.append(action)
                # reduce
                if action in action_id["LR"] or action in action_id["NR"]:
                    stack.pop(-1)
                    stack_len -= 1

                # pass
                elif action in action_id["LP"] or action in action_id[
                        "NP"] or action in action_id["RP"]:
                    stack_top = stack.pop(-1)
                    stack_len -= 1
                    deque.append(stack_top)

                # shift
                elif action in action_id["RS"] or action in action_id["NS"]:
                    j = len(deque) - 1
                    start = j
                    while (deque[j] != seq_len):
                        stack.append(deque.pop(-1))
                        j -= 1
                    stack_len += start - j

                    buffer_top = buffer.pop(0)
                    stack.append(buffer_top)
                    stack_len += 1
                    buffer_len -= 1

            loss = torch.sum(torch.stack(loss)) / (len(loss) + 1)
            losses.append(loss)

        losses = torch.sum(torch.stack(losses)) / batch_size
        return losses

    def forward(self, words, feats, states, gold_actions, transition_len):
        r'''
        Args:
            words: [batch_size, seq_len]
            feats:
            states:[batch_size, padded_transition_len, 4, just_len]
                    padded_transition_len=padded_state_len
            gold_actions:[batch_size, padded_transition_len]
                    just for train the label predict
            window: the num of cell to look in stack, deque, buffer, transition_history
        '''

        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)

        feat_embeds = []
        if 'tag' in self.args.feat:
            feat_embeds.append(self.tag_embed(feats.pop()))
        if 'char' in self.args.feat:
            feat_embeds.append(self.char_embed(feats.pop(0)))
        if 'bert' in self.args.feat:
            feat_embeds.append(self.bert_embed(feats.pop(0)))
        if 'lemma' in self.args.feat:
            feat_embeds.append(self.lemma_embed(feats.pop(0)))
        word_embed, feat_embed = self.embed_dropout(word_embed,
                                                    torch.cat(feat_embeds, -1))
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        x = pack_padded_sequence(embed, mask.sum(1), True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)
        # x:[batch_size, seq_len, hidden_size*2]
        null_hidden = self.null_lstm_hidden.expand(batch_size, -1, -1)
        x = torch.cat([x, null_hidden], dim=1)
        # x:[batch_size, seq_len+1, hidden_size*2]
        lstm_out_size = x.shape[2]

        # 这一部分可以考虑移出去
        windowed = states[..., :self.window]
        # [batch_size, padded_transition_len, 4, window]
        win_states = windowed[:, :, 0:3, :]
        null_lstm_mask = win_states.eq(-1)
        win_states = win_states.masked_fill(null_lstm_mask, seq_len)
        # [batch_size, padded_transition_len, 3, window]

        bs, padded_transition_len, s2, s3 = win_states.shape
        win_states = win_states.reshape(bs, padded_transition_len * s2 *
                                        s3).unsqueeze(-1).expand(
                                            -1, -1, lstm_out_size)
        states_hidden = torch.gather(x, 1, win_states).reshape(
            bs, padded_transition_len, s2, s3, -1)
        # [batch_size, padded_transition_len, 3, window, lstm_out_size]
        states_hidden = torch.sum(states_hidden,
                                  dim=3).reshape(batch_size,
                                                 padded_transition_len, -1)
        # [batch_size, padded_transition_len, 3*lstm_out_size]

        transitions = windowed[:, :, 3:, :].squeeze(
            2)  # [batch_size, padded_transition_len, window]
        transitions_mask = transitions.eq(-1)
        transitions = transitions.masked_fill(transitions_mask,
                                              self.n_transitions)
        transitions = transitions.reshape(batch_size,
                                          -1)  # [batch_size, pad_len*window]

        # transitions_embed = self.transition_embed_droupout(
        #     self.transition_embed(transitions))[0]
        transitions_embed = self.new_droupout(
            self.transition_embed(transitions))

        transitions_embed = transitions_embed.reshape(batch_size,
                                                      padded_transition_len,
                                                      self.window, -1)

        transitions_embed = torch.sum(
            transitions_embed,
            dim=2)  # [batch_size, pad_len, n_transition_embed]

        if (self.decode_mode != 'beta'):
            final_repr = torch.cat([transitions_embed, states_hidden], -1)
            # [batch_size, pad_len, n_transition_embed+3*lstm_out_size]
        else:
            states_hidden = states_hidden.reshape(batch_size,
                                                  padded_transition_len, 3, -1)
            inter = self.inter.reshape(3, 1).expand(-1, states_hidden.shape[3])
            intered = states_hidden * inter
            intered = torch.sum(intered, dim=2)
            # [batch_size, padded_transition_len, lstm_out_size]
            final_repr = torch.cat([transitions_embed, intered], -1)
            # [batch_size, padded_transition_len, n_transition_embed + lstm_out_size]
            score = self.mlp(final_repr)
            return score

        if (self.decode_mode == 'mlp'):
            score = self.base_mlp(final_repr)

        elif (self.decode_mode == 'att'):
            states_repr = self.state_mlp(final_repr)
            # [batch_size, pad_len, n_mlp]
            transitions_repr = self.transition_embed(
                torch.tensor([i for i in range(self.n_transitions)],
                             device=states_repr.device))
            transitions_mlp = self.action_mlp(transitions_repr)
            transitions_final = transitions_mlp.transpose(1, 0)
            # [n_mlp, n_transitions]
            score = torch.matmul(states_repr, transitions_final)
            # [batch_size, pad_len, n_transitions]

        elif (self.decode_mode == 'lstm'):
            lstm_in = pack_padded_sequence(final_repr, transition_len, True,
                                           False)
            lstm_out, _ = self.decoder_lstm(lstm_in)
            lstm_out, _ = pad_packed_sequence(
                lstm_out, True, total_length=padded_transition_len)
            lstm_out = self.decoder_lstm_dropout(lstm_out)
            # lstm_out:[batch_size, pad_len, n_lstm_hidden]
            score = self.decoder_lstm_mlp(lstm_out)

        elif (self.decode_mode == 'dual'):
            action_score = self.action_mlp(final_repr)
            # [batch_size, pad_len, n_transitions]
            gold_mask = gold_actions.eq(-1)
            ga = gold_actions.masked_fill(gold_mask, self.n_transitions)
            ga_embed = self.transition_embed(ga)
            # [batch_size, pad_len, n_transition_embed]
            label_final = torch.cat((final_repr, ga_embed), -1)
            label_score = self.label_mlp(label_final)
            return action_score, label_score

        else:
            print('error decoder mode')
            exit(-1)
        return score

    def loss(self, score, gold_transitions):
        r"""
        Args:
            score: [batch_size, padded_transition_len, n_transition]
            gold_transitions: [batch_size, padded_transition_len]

        Returns:
            ~torch.Tensor:
                The training loss.
        """

        mask = gold_transitions.ge(0)
        loss = self.criterion(score[mask], gold_transitions[mask])
        return loss

    def dual_loss(self, action_score, label_score, gold_action, gold_label):
        r"""
        Args:
            action_score: [batch_size, padded_transition_len, n_transition]
            gold_action: [batch_size, padded_transition_len]
            label_score: [batch_size, padded_transition_len, n_labels+1]
            gold_label: [batch_size, padded_transition_len]

        Returns:
            ~torch.Tensor:
                The training loss.
        """
        mask = gold_action.ge(0)
        # pdb.set_trace()
        action_loss = self.criterion(action_score[mask], gold_action[mask])
        label_loss = self.criterion(label_score[mask], gold_label[mask])
        return action_loss + label_loss

    
    def decode(self, words, words_len, feats):
        r"""
        words: [batch_size, seq_len]
        words_len: [batch_size]
        """
        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)

        feat_embeds = []
        if 'tag' in self.args.feat:
            feat_embeds.append(self.tag_embed(feats.pop()))
        if 'char' in self.args.feat:
            feat_embeds.append(self.char_embed(feats.pop(0)))
        if 'bert' in self.args.feat:
            feat_embeds.append(self.bert_embed(feats.pop(0)))
        if 'lemma' in self.args.feat:
            feat_embeds.append(self.lemma_embed(feats.pop(0)))
        word_embed, feat_embed = word_embed, torch.cat(feat_embeds, -1)
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        x = pack_padded_sequence(embed, mask.sum(1), True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        # x:[batch_size, seq_len, hidden_size*2]
        null_hidden = self.null_lstm_hidden.expand(batch_size, -1, -1)
        x = torch.cat([x, null_hidden], dim=1)
        # x:[batch_size, seq_len+1, hidden_size*2]
        lstm_out_size = x.shape[2]

        arcs = []
        labels = []

        action_id = {
            action_: [
                self.transition_vocab.stoi[a]
                for a in self.transition_vocab.stoi.keys()
                if a.startswith(action_)
            ]
            for action_ in ["LR", "LP", "RS", "RP", "NS", "NR", "NP"]
        }

        for i in range(batch_size):
            look_up = x[i]
            arc = [[0] * seq_len for k in range(seq_len)]
            label = [[-1] * seq_len for k in range(seq_len)]

            actions = [self.n_transitions] * self.window  # 动作序列,-1头
            stack = [seq_len] * self.window + [0]  # 正在处理的节点序列,-1头
            buffer = [k for k in range(1, words_len[i].item())
                      ] + [seq_len] * self.window  # 待处理的节点序列,0头
            deque = [seq_len] * self.window  # 暂时跳过的节点序列（可能存在多个头节点）,-1头
            buffer_len = words_len[i].item() - 1
            stack_len = 1
            count = 0

            h_t = torch.zeros(
                1, 1,
                (self.args.n_lstm_hidden * 6 + self.args.n_transition_embed) //
                3).to(x.device)
            c_t = torch.zeros(
                1, 1,
                (self.args.n_lstm_hidden * 6 + self.args.n_transition_embed) //
                3).to(x.device)

            while (buffer_len > 0):
                if (count > 700):
                    print('count supass 500')
                    break
                count += 1

                valid_actions = []

                if (stack_len > 1 and buffer_len > 0):
                    valid_actions += action_id['LR']
                    valid_actions += action_id['LP']
                    valid_actions += action_id['RP']

                if (buffer_len > 0):
                    valid_actions += action_id['NS']
                    if (stack_len > 0):
                        valid_actions += action_id['RS']  # ROOT,NULL

                if (stack_len > 1):
                    valid_actions += action_id['NR']
                    valid_actions += action_id['NP']

                action = valid_actions[0]
                final_repr = None
                if len(valid_actions) > 1:
                    stack_embed = torch.sum(look_up[stack[-self.window:]],
                                            dim=0)
                    buffer_embed = torch.sum(look_up[buffer[0:self.window]],
                                             dim=0)
                    deque_embed = torch.sum(look_up[deque[-self.window:]],
                                            dim=0)
                    transition_embed = torch.sum(self.transition_embed(
                        torch.tensor(actions[-self.window:],
                                     dtype=torch.long,
                                     device=x.device)),
                                                 dim=0)
                    final_repr = torch.cat((transition_embed, stack_embed,
                                            buffer_embed, deque_embed),
                                           dim=-1).unsqueeze(0)

                    if (self.decode_mode == 'mlp'):
                        score = self.base_mlp(final_repr).squeeze(0)[
                            torch.tensor(valid_actions,
                                         dtype=torch.long,
                                         device=x.device)]
                    elif (self.decode_mode == 'att'):

                        # [1,n_transition_embed+3*lstm_out_size]
                        states_repr = self.state_mlp(final_repr)
                        # [1, n_mlp]
                        transitions_repr = self.transition_embed(
                            torch.tensor(
                                [i for i in range(self.n_transitions)],
                                device=states_repr.device))
                        transitions_final = self.action_mlp(
                            transitions_repr).transpose(1, 0)
                        # [n_mlp, n_transitions]
                        score = torch.matmul(
                            states_repr,
                            transitions_final).squeeze(0)[torch.tensor(
                                valid_actions,
                                dtype=torch.long,
                                device=x.device)]

                    elif (self.decode_mode == 'lstm'):

                        # [1,n_transition_embed+3*lstm_out_size]
                        final_repr = final_repr.unsqueeze(0)
                        out, (h_t,
                              c_t) = self.decoder_lstm(final_repr, (h_t, c_t))
                        out = out.squeeze(0)
                        score = self.decoder_lstm_mlp(out).squeeze(0)[
                            torch.tensor(valid_actions,
                                         dtype=torch.long,
                                         device=x.device)]

                    elif (self.decode_mode == 'beta'):
                        plus = torch.stack(
                            (stack_embed, buffer_embed, deque_embed))
                        inter = self.inter.reshape(3, 1).expand(
                            -1, plus.shape[1])
                        plus = plus * inter
                        plus = torch.sum(plus, dim=0)
                        # [lstm_hiddem*2]
                        final_repr = torch.cat((transition_embed, plus),
                                               -1).unsqueeze(0)
                        score = self.mlp(final_repr).squeeze(0)[torch.tensor(
                            valid_actions, dtype=torch.long, device=x.device)]

                    elif (self.decode_mode == 'dual'):
                        score = self.action_mlp(final_repr).squeeze(0)[
                            torch.tensor(valid_actions,
                                         dtype=torch.long,
                                         device=x.device)]

                    action_idx = torch.argmax(score, dim=0).item()
                    action = valid_actions[action_idx]

                #  现在这个第二阶段是依赖于第一阶段的，之后或许可以改成独立的
                if action in action_id["LR"] or action in action_id["LP"] or \
                        action in action_id["RS"] or action in action_id["RP"]:
                    if action in action_id["RS"] or action in action_id["RP"]:
                        head = stack[-1]
                        modifier = buffer[0]
                    else:
                        head = buffer[0]
                        modifier = stack[-1]

                    arc[modifier][head] = 1
                    if (self.decode_mode != 'dual'):
                        label[modifier][head] = int(
                            self.transition_vocab.itos[action].split(':')[1])
                    else:
                        if (final_repr is None):
                            stack_embed = torch.sum(
                                look_up[stack[-self.window:]], dim=0)
                            buffer_embed = torch.sum(
                                look_up[buffer[0:self.window]], dim=0)
                            deque_embed = torch.sum(
                                look_up[deque[-self.window:]], dim=0)
                            transition_embed = torch.sum(self.transition_embed(
                                torch.tensor(actions[-self.window:],
                                             dtype=torch.long,
                                             device=x.device)),
                                                         dim=0)
                            final_repr = torch.cat(
                                (transition_embed, stack_embed, buffer_embed,
                                 deque_embed),
                                dim=-1).unsqueeze(0)

                        current_action_embed = self.transition_embed(
                            torch.tensor([action],
                                         dtype=torch.long,
                                         device=x.device))
                        label_final = torch.cat(
                            (final_repr, current_action_embed), -1)
                        label_score = self.label_mlp(label_final).squeeze(0)[
                            torch.tensor([vla for vla in range(self.n_labels)],
                                         dtype=torch.long,
                                         device=x.device)]
                        label_idx = torch.argmax(label_score, dim=0).item()
                        label[modifier][head] = label_idx

                actions.append(action)

                # reduce
                if action in action_id["LR"] or action in action_id["NR"]:
                    stack.pop(-1)
                    stack_len -= 1

                # pass
                elif action in action_id["LP"] or action in action_id[
                        "NP"] or action in action_id["RP"]:
                    stack_top = stack.pop(-1)
                    stack_len -= 1
                    deque.append(stack_top)

                # shift
                elif action in action_id["RS"] or action in action_id["NS"]:
                    j = len(deque) - 1
                    start = j
                    while (deque[j] != seq_len):
                        stack.append(deque.pop(-1))
                        j -= 1
                    stack_len += start - j

                    buffer_top = buffer.pop(0)
                    stack.append(buffer_top)
                    stack_len += 1
                    buffer_len -= 1

            arcs.append(arc)
            labels.append(label)

        return torch.tensor(arcs, device=words.device), torch.tensor(
            labels, device=words.device)

    def dynamic_loss2(self, words, words_len, feats, edges, labels):
        r"""
        目前的explore处理的粒度是整个batch，之后应该会试试step粒度的explore
        目前只适用dual
        目前的优化目标只是Correct(c)中分数最高的
        words: [batch_size, seq_len]
        words_len: [batch_size]
        edges : [batch_size, seq_len, seq_len]
        labels: [batch_size, seq_len, seq_len]
        
        对图中没有的边就不加到correct中（有可能会造成correct为空)
        """

        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)

        feat_embeds = []
        if 'tag' in self.args.feat:
            feat_embeds.append(self.tag_embed(feats.pop()))
        if 'char' in self.args.feat:
            feat_embeds.append(self.char_embed(feats.pop(0)))
        if 'bert' in self.args.feat:
            feat_embeds.append(self.bert_embed(feats.pop(0)))
        if 'lemma' in self.args.feat:
            feat_embeds.append(self.lemma_embed(feats.pop(0)))
        word_embed, feat_embed = self.embed_dropout(word_embed,
                                                    torch.cat(feat_embeds, -1))
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        x = pack_padded_sequence(embed, mask.sum(1), True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)
        # x:[batch_size, seq_len, hidden_size*2]
        null_hidden = self.null_lstm_hidden.expand(batch_size, -1, -1)
        x = torch.cat([x, null_hidden], dim=1)
        # x:[batch_size, seq_len+1, hidden_size*2]
        lstm_out_size = x.shape[2]

        action_id = {
            action_: [
                self.transition_vocab.stoi[a]
                for a in self.transition_vocab.stoi.keys()
                if a.startswith(action_)
            ]
            for action_ in ["LR", "LP", "RS", "RP", "NS", "NR", "NP"]
        }

        losses = []
        for i in range(batch_size):
            look_up = x[i]
            edge = edges[i].cpu().numpy()
            label = labels[i].cpu().numpy()
            loss = []

            # arc = [[0] * seq_len for k in range(seq_len)]
            # label = [[-1] * seq_len for k in range(seq_len)]

            actions = [self.n_transitions] * self.window  # 动作序列,-1头
            stack = [seq_len] * self.window + [0]  # 正在处理的节点序列,-1头
            buffer = [k for k in range(1, words_len[i].item())
                      ] + [seq_len] * self.window  # 待处理的节点序列,0头
            deque = [seq_len] * self.window  # 暂时跳过的节点序列（可能存在多个头节点）,-1头
            buffer_len = words_len[i].item() - 1
            stack_len = 1
            count = 0

            while (buffer_len > 0):
                if (count > 500):
                    print('count supass 500')
                    break
                count += 1

                valid_actions = []
                if (stack_len > 1 and buffer_len > 0):
                    valid_actions += action_id['LR']
                    valid_actions += action_id['LP']
                    valid_actions += action_id['RP']

                if (buffer_len > 0):
                    valid_actions += action_id['NS']
                    if (stack_len > 0):
                        valid_actions += action_id['RS']  # ROOT,NULL

                if (stack_len > 1):
                    valid_actions += action_id['NR']
                    valid_actions += action_id['NP']

                correct_actions = []  #用来存储在当前情况下，cost为0的transition
                correct_labels = []  #用来存储在当前情况下，正确transition对应的label序列
                stack_head = stack[-1]
                buffer_head = buffer[0]
                # 这个地方可能要判断stack_head是否是seq_len
                if (stack_head == seq_len):
                    correct_actions += action_id['NS']
                    correct_labels.append([self.n_labels])
                else:
                    flag = 1
                    if (edge[buffer_head][stack_head]):
                        flag = 0
                    else:
                        for b in buffer[1:-self.window]:
                            if (edge[stack_head][b] or edge[b][stack_head]):
                                flag = 0
                                break
                    if (flag):
                        # LR这个action是可以的,但是还没考虑label
                        if (label[stack_head][buffer_head] != -1):
                            correct_labels.append(
                                [label[stack_head][buffer_head]])
                            correct_actions += action_id['LR']

                    flag = 1
                    if (edge[stack_head][buffer_head]):
                        flag = 0
                    else:
                        for s in stack[self.window:-1]:
                            if (edge[s][buffer_head] or edge[buffer_head][s]):
                                flag = 0
                                break
                    if (flag):
                        # RS
                        if (label[buffer_head][stack_head] != -1):
                            correct_labels.append(
                                [label[buffer_head][stack_head]])
                            correct_actions += action_id['RS']

                    flag = 1
                    for a in stack[self.window:]:
                        if (edge[a][buffer_head] or edge[buffer_head][a]):
                            flag = 0
                            break
                    if (flag):
                        # NS
                        correct_actions += action_id['NS']
                        correct_labels.append([self.n_labels])  #加入空标签对应的下标

                    flag = 1
                    for b in buffer[:-self.window]:
                        if (edge[b][stack_head] or edge[stack_head][b]):
                            flag = 0
                            break
                    if (flag):
                        # NR
                        correct_actions += action_id['NR']
                        correct_labels.append([self.n_labels])

                    flag = 1
                    if (edge[buffer_head][stack_head]):
                        flag = 0
                    if (flag):
                        # LP
                        if (label[stack_head][buffer_head] != -1):
                            correct_labels.append(
                                [label[stack_head][buffer_head]])
                            correct_actions += action_id['LP']

                    flag = 1
                    if (edge[stack_head][buffer_head]):
                        flag = 0
                    if (flag):
                        # RP
                        if (label[buffer_head][stack_head] != -1):
                            correct_labels.append(
                                [label[buffer_head][stack_head]])
                            correct_actions += action_id['RP']

                    flag = 1
                    if (edge[buffer_head][stack_head]
                            or edge[stack_head][buffer_head]):
                        flag = 0
                    if (flag):
                        # NP
                        correct_actions += action_id['NP']
                        correct_labels.append([self.n_labels])

                correct_idx = [
                    idx for idx in range(len(correct_actions))
                    if correct_actions[idx] in valid_actions
                ]
                # print('idxs', correct_idx)
                # print('actions', correct_actions)
                # print('labels', correct_labels)
                correct_actions_ = [
                    correct_actions[idx] for idx in correct_idx
                ]
                correct_labels_ = [correct_labels[idx] for idx in correct_idx]
                # 怀疑这个correct_actions可能为空

                action = valid_actions[0]
                o_action = correct_actions_[0]
                o_labels = correct_labels_[0]

                stack_embed = torch.sum(look_up[stack[-self.window:]], dim=0)
                buffer_embed = torch.sum(look_up[buffer[0:self.window]], dim=0)
                deque_embed = torch.sum(look_up[deque[-self.window:]], dim=0)
                last_t_embed = self.transition_embed(
                    torch.tensor(actions[-self.window:],
                                 dtype=torch.long,
                                 device=x.device))
                # last_t_embed_droup = self.transition_embed_droupout(last_t_embed)[0]
                last_t_embed_droup = self.new_droupout(last_t_embed)
                transition_embed = torch.sum(last_t_embed_droup, dim=0)
                # pdb.set_trace()
                final_repr = torch.cat(
                    (transition_embed, stack_embed, buffer_embed, deque_embed),
                    dim=-1).unsqueeze(0)

                score = self.action_mlp(final_repr)

                if (len(valid_actions) > 1):
                    action_idx = torch.argmax(score.squeeze(0)[torch.tensor(
                        valid_actions, dtype=torch.long, device=x.device)],
                                              dim=0).item()
                    action = valid_actions[action_idx]

                if (len(correct_actions_) > 1):
                    o_action_idx = torch.argmax(score.squeeze(0)[torch.tensor(
                        correct_actions_, dtype=torch.long, device=x.device)],
                                                dim=0).item()
                    o_action = correct_actions_[o_action_idx]
                    o_labels = correct_labels_[o_action_idx]

                action_loss = self.criterion(
                    score,
                    torch.tensor([o_action],
                                 dtype=torch.long,
                                 device=score.device))

                current_action_embed = self.transition_embed(
                    torch.tensor([action], dtype=torch.long, device=x.device))
                label_final = torch.cat((final_repr, current_action_embed), -1)
                label_score = self.label_mlp(label_final)

                if (len(o_labels) > 1):
                    label_idx = torch.argmax(
                        label_score.squeeze(0)[torch.tensor(
                            o_labels,
                            dtype=torch.long,
                            device=label_score.device)],
                        dim=0).item()
                    y_label = o_labels[label_idx]
                else:
                    y_label = o_labels[0]

                label_loss = self.criterion(
                    label_score,
                    torch.tensor([y_label],
                                 dtype=torch.long,
                                 device=label_score.device))

                loss_step = label_loss + action_loss
                loss.append(loss_step)

                actions.append(action)
                # reduce
                if action in action_id["LR"] or action in action_id["NR"]:
                    stack.pop(-1)
                    stack_len -= 1

                # pass
                elif action in action_id["LP"] or action in action_id[
                        "NP"] or action in action_id["RP"]:
                    stack_top = stack.pop(-1)
                    stack_len -= 1
                    deque.append(stack_top)

                # shift
                elif action in action_id["RS"] or action in action_id["NS"]:
                    j = len(deque) - 1
                    start = j
                    while (deque[j] != seq_len):
                        stack.append(deque.pop(-1))
                        j -= 1
                    stack_len += start - j

                    buffer_top = buffer.pop(0)
                    stack.append(buffer_top)
                    stack_len += 1
                    buffer_len -= 1

            loss = torch.sum(torch.stack(loss)) / (len(loss) + 1)
            losses.append(loss)

        losses = torch.sum(torch.stack(losses)) / batch_size
        return losses

    def dynamic_loss3(self, words, words_len, feats, edges, labels, p):
        r"""
        目前只适用dual
        目前的优化目标只是Correct(c)中分数最高的
        words: [batch_size, seq_len]
        words_len: [batch_size]
        edges : [batch_size, seq_len, seq_len]
        labels: [batch_size, seq_len, seq_len]
        还是只保证尽可能多去get到gold里面的边，但是有可能会生成错误的边，这一点之后可能会导致
        召回率高，但是准确率低.
        batch化
        """

        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)

        feat_embeds = []
        if 'tag' in self.args.feat:
            feat_embeds.append(self.tag_embed(feats.pop()))
        if 'char' in self.args.feat:
            feat_embeds.append(self.char_embed(feats.pop(0)))
        if 'bert' in self.args.feat:
            feat_embeds.append(self.bert_embed(feats.pop(0)))
        if 'lemma' in self.args.feat:
            feat_embeds.append(self.lemma_embed(feats.pop(0)))
        word_embed, feat_embed = self.embed_dropout(word_embed,
                                                    torch.cat(feat_embeds, -1))
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        x = pack_padded_sequence(embed, mask.sum(1), True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)
        # x:[batch_size, seq_len, hidden_size*2]
        null_hidden = self.null_lstm_hidden.expand(batch_size, -1, -1)
        x = torch.cat([x, null_hidden], dim=1)
        # x:[batch_size, seq_len+1, hidden_size*2]
        lstm_out_size = x.shape[2]

        sdt = torch.tensor([[0] + [-1] *
                            (seq_len - 1), [-1] * seq_len, [-1] * seq_len],
                           dtype=torch.long,
                           device=words.device)
        # 这边先用-1pad，之后取的时候，要先fill，stack,buffer,deque用seq_len,pre_action用n_transitions
        sdt = sdt.unsqueeze(0).expand(batch_size, -1, -1)
        bu_lst = []
        for length in words_len:
            bu_lst.append(
                torch.arange(1,
                             length.item(),
                             dtype=torch.long,
                             device=words.device))
        bu_out = pad(bu_lst, -1,
                     seq_len).to(words.device).reshape(batch_size, 1, -1)
        init_stat = torch.cat((sdt, bu_out), dim=1)
        save_stat = init_stat[:, [0, 3, 2, 1]]
        # 初始状态:[batch _size, 4, seq_len] stack,buffer,deque,pre_action
        # pad with -1
        # remain_edge = edges
        # remain_label = labels
        # remain_x = x
        unfinish_mask = save_stat[:, 1, 0].gt(-1)

        action_id = {
            action_: [
                self.transition_vocab.stoi[a]
                for a in self.transition_vocab.stoi.keys()
                if a.startswith(action_)
            ]
            for action_ in ["LR", "LP", "RS", "RP", "NS", "NR", "NP"]
        }
        
        losses = []
        remain_ks = []
        step = 0
        while (unfinish_mask.sum()):
            step += 1
            remain_stat = save_stat[unfinish_mask]
            # [k, 4, seq_len]
            remain_edge = edges[unfinish_mask]
            remain_label = labels[unfinish_mask]
            # [k, seq_len, seq_len]
            remain_x = x[unfinish_mask]
            # [k, seq_len+1, lstm_out_size]
            k = remain_stat.shape[0]
            remain_ks.append(k/batch_size)

            valid1 = torch.ones((k, len(self.transition_vocab)),
                                dtype=torch.long,
                                device=words.device)
            base_mask = valid1.gt(0)
            valid1_mask = base_mask.clone()
            stat_len = remain_stat.gt(-1).sum(dim=2)
            # [k, 4]
            valid_mask_sg1 = stat_len[:, 0].gt(1)
            # [k]
            valid_mask_bg0 = stat_len[:, 1].gt(0)
            # [k]
            valid_mask_sg0 = stat_len[:, 0].gt(0)
            # [k]
            mask_v1 = valid_mask_sg1 & valid_mask_bg0
            mask_v1 = mask_v1.unsqueeze(1)
            valid1_mask = valid1_mask & mask_v1
            col_cond1 = torch.zeros(
                len(self.transition_vocab), device=words.device).index_fill_(
                    -1,
                    torch.tensor(action_id['LR'] + action_id['LP'] +
                                 action_id['RP'],
                                 device=words.device), 1).gt(0)
            valid1_mask = valid1_mask * col_cond1
            valid1.masked_fill_(~valid1_mask, 0)

            valid2 = torch.ones((k, len(self.transition_vocab)),
                                dtype=torch.long,
                                device=words.device)
            valid2_mask = base_mask.clone()

            mask_v2 = valid_mask_bg0.unsqueeze(1)
            valid2_mask = valid2_mask & mask_v2
            col_cond2 = torch.zeros(len(self.transition_vocab),
                                    device=words.device).index_fill_(
                                        -1,
                                        torch.tensor(action_id['NS'],
                                                     device=words.device),
                                        1).gt(0)
            valid2_mask = valid2_mask * col_cond2
            valid2.masked_fill_(~valid2_mask, 0)

            valid3 = torch.ones((k, len(self.transition_vocab)),
                                dtype=torch.long,
                                device=words.device)
            valid3_mask = base_mask.clone()
            mask_v3 = valid_mask_bg0 & valid_mask_sg0
            mask_v3 = mask_v3.unsqueeze(1)
            valid3_mask = valid3_mask & mask_v3
            col_cond3 = torch.zeros(len(self.transition_vocab),
                                    device=words.device).index_fill_(
                                        -1,
                                        torch.tensor(action_id['RS'],
                                                     device=words.device),
                                        1).gt(0)
            valid3_mask = valid3_mask * col_cond3
            valid3.masked_fill_(~valid3_mask, 0)

            valid4 = torch.ones((k, len(self.transition_vocab)),
                                dtype=torch.long,
                                device=words.device)
            valid4_mask = base_mask.clone()
            msak_v4 = valid_mask_sg1.unsqueeze(1)
            valid4_mask = valid4_mask & msak_v4
            col_cond4 = torch.zeros(len(self.transition_vocab),
                                    device=words.device).index_fill_(
                                        -1,
                                        torch.tensor(
                                            action_id['NR'] + action_id['NP'],
                                            device=words.device), 1).gt(0)
            valid4_mask = valid4_mask * col_cond4
            valid4.masked_fill_(~valid4_mask, 0)
            valid_actions = valid1 + valid2 + valid3 + valid4
            # pdb.set_trace()
            # [k, 7]

            # 求correct集合
            correct_actions = torch.zeros((k, len(self.transition_vocab)),
                                          dtype=torch.long,
                                          device=words.device)
            # [k, 7]
            correct_labels = torch.ones(
                (k, len(self.transition_vocab)),
                dtype=torch.long,
                device=words.device).masked_fill_(base_mask, self.n_labels)
            # 暂时correct labels存的还是一个，不是多个，对应less correct的版本

            stack_isnot_clear = valid_mask_sg0.clone()
            # [k]
            stack_is_clear = ~stack_isnot_clear.unsqueeze(1)
            ns_mask = base_mask & stack_is_clear
            col_ns_mask = torch.zeros(len(self.transition_vocab),
                                      device=words.device).index_fill_(
                                          -1,
                                          torch.tensor(action_id['NS'],
                                                       device=words.device),
                                          1).gt(0)
            ns_mask = ns_mask * col_ns_mask
            correct_actions.masked_fill_(ns_mask, 1)
            # correct_labels里面存的就是self.n_labels，所以不要动

            stack_isnot_clear_idx = (
                stack_isnot_clear.int() == 1).nonzero().squeeze(1)
            # [m] 保存的是k个里面stack非空的那些的下标
            if (stack_isnot_clear_idx.shape[0] > 0):
                stack_exist_stat = remain_stat[stack_isnot_clear]
                m = stack_exist_stat.shape[0]
                # [m, 4, seq_len]
                m_actions = torch.zeros((m, len(self.transition_vocab)),
                                        dtype=torch.long,
                                        device=words.device)
                m_labels = torch.ones(
                    (m, len(self.transition_vocab)),
                    dtype=torch.long,
                    device=words.device).masked_fill_(base_mask, self.n_labels)

                stack_exist_edge = remain_edge[stack_isnot_clear]
                stack_exist_label = remain_label[stack_isnot_clear]
                # [m, seq_len, seq_len]

                exist_right = stack_exist_edge[:, stack_exist_stat[:, 1, 0],
                                               stack_exist_stat[:, 0, 0]].diag(
                                                   0).gt(0)
                # [m] 有没有从i指向j的边（右边）
                exist_left = stack_exist_edge[:, stack_exist_stat[:, 0, 0],
                                              stack_exist_stat[:, 1, 0]].diag(
                                                  0).gt(0)
                # [m] 有没有从j指向i的边（左边）
                left_label = stack_exist_label[:, stack_exist_stat[:, 0, 0],
                                               stack_exist_stat[:, 1, 0]].diag(
                                                   0).clone()
                left_label = left_label.masked_fill(left_label.eq(-1),
                                                    self.n_labels)
                # [m] 所有左边的label
                right_label = stack_exist_label[:, stack_exist_stat[:, 1, 0],
                                                stack_exist_stat[:, 0,
                                                                 0]].diag(
                                                                     0).clone(
                                                                     )
                right_label = right_label.masked_fill(right_label.eq(-1),
                                                      self.n_labels)
                # [m] 所有的右边label

                stack_head = stack_exist_stat[:, 0, 0].clone()
                # [m]
                sh1 = stack_head.unsqueeze(1).expand(-1, seq_len).unsqueeze(1)
                # [m, 1, seq_len]
                sh2 = stack_head.unsqueeze(1).expand(-1, seq_len - 1)
                condicate_b = stack_exist_stat[:, 1, 1:].clone()
                # [m, seq_len-1]
                g1 = torch.gather(stack_exist_edge, 1, sh1)
                # [m, 1, seq_len]
                condicate_b_mask = condicate_b.eq(-1)
                condicate_b_temp = condicate_b.masked_scatter(
                    condicate_b_mask, sh2[condicate_b_mask]).unsqueeze(1)
                # [m, 1, seq_len-1]
                g2 = torch.gather(g1, 2, condicate_b_temp).squeeze(1)
                # [m, seq_len-1]
                if_i_beta = g2.sum(1).gt(0)
                # [m]  有没有从beta指向i（栈顶）的边

                sh1 = sh1.expand(-1, seq_len - 1, -1)
                # [m, seq_len-1, seq_len]
                m1 = condicate_b.unsqueeze(2).expand(-1, -1, seq_len)
                # [m, seq_len-1, seq_len]
                m1_mask = m1.eq(-1)
                m1 = m1.masked_scatter(m1_mask, sh1[m1_mask])
                m2 = torch.gather(stack_exist_edge, 1, m1)
                # [m, seq_len-1, seq_len]
                sh2 = stack_head.unsqueeze(1).unsqueeze(1).expand(
                    -1, seq_len - 1, -1)
                # [m, seq_len-1, 1]
                m3 = torch.gather(m2, 2, sh2).squeeze(2)
                # [m, seq_len-1]
                if_beta_i = m3.sum(1).gt(0)
                # [m] 有没有从i指向beta的边
                or_beta_i = if_beta_i + if_i_beta
                final_lr = ~exist_right & ~or_beta_i
                # [m]  不存在向右的边 and i与beta之间没有边
                final_lr_1 = final_lr.unsqueeze(1).expand(
                    -1, len(self.transition_vocab))
                # [m, n_actions]
                col_lr = torch.zeros(len(self.transition_vocab),
                                     device=words.device).index_fill_(
                                         -1,
                                         torch.tensor(action_id['LR'],
                                                      device=words.device),
                                         1).gt(0)
                lr_mask = final_lr_1 * col_lr
                # [m, n_actions]
                m_actions.masked_fill_(lr_mask, 1)
                m_labels = m_labels.masked_scatter(lr_mask,
                                                   left_label[final_lr])

                buffer_head = stack_exist_stat[:, 1, 0].clone()
                sh1 = buffer_head.unsqueeze(1).expand(-1, seq_len).unsqueeze(1)
                sh2 = buffer_head.unsqueeze(1).expand(-1, seq_len - 1)
                condicate_s = stack_exist_stat[:, 0, 1:].clone()
                g1 = torch.gather(stack_exist_edge, 1, sh1)
                condicate_s_mask = condicate_s.eq(-1)
                condicate_s_temp = condicate_s.masked_scatter(
                    condicate_s_mask, sh2[condicate_s_mask]).unsqueeze(1)
                g2 = torch.gather(g1, 2, condicate_s_temp).squeeze(1)
                if_j_sigma = g2.sum(1).gt(0)
                # [m] 有没有从sigma指向j的边

                sh1 = sh1.expand(-1, seq_len-1, -1)
                m1 = condicate_s.unsqueeze(2).expand(-1, -1, seq_len)
                m1_mask = m1.eq(-1)
                m1 = m1.masked_scatter(m1_mask, sh1[m1_mask])
                m2 = torch.gather(stack_exist_edge, 1, m1)
                sh2 = buffer_head.unsqueeze(1).unsqueeze(1).expand(
                    -1, seq_len - 1, -1)
                m3 = torch.gather(m2, 2, sh2).squeeze(2)
                if_sigma_j = m3.sum(1).gt(0)
                # [m] 有没有从j指向sigma的边
                or_sigma_j = if_j_sigma + if_sigma_j
                final_rs = ~exist_left & ~or_sigma_j
                # [m] 不存在向左边的边 且 j与sigma之间没有边
                final_rs_1 = final_rs.unsqueeze(1).expand(
                    -1, len(self.transition_vocab))
                col_rs = torch.zeros(len(self.transition_vocab),
                                     device=words.device).index_fill_(
                                         -1,
                                         torch.tensor(action_id['RS'],
                                                      device=words.device),
                                         1).gt(0)
                rs_mask = final_rs_1 * col_rs
                m_actions.masked_fill_(rs_mask, 1)
                m_labels = m_labels.masked_scatter(rs_mask,
                                                   right_label[final_rs])

                if_stack_j = exist_left + if_sigma_j
                # [m] 有没有从j指向stack的边
                if_j_stack = exist_right + if_j_sigma
                # [m] 有没有从stack指向j的边
                or_stack_j = if_stack_j + if_j_stack
                # [m] stack与j之间有没有边
                final_ns = ~or_stack_j.unsqueeze(1).expand(
                    -1, len(self.transition_vocab))
                col_ns = torch.zeros(len(self.transition_vocab),
                                     device=words.device).index_fill_(
                                         -1,
                                         torch.tensor(action_id['NS'],
                                                      device=words.device),
                                         1).gt(0)
                Ns_mask = final_ns * col_ns
                m_actions.masked_fill_(Ns_mask, 1)
                # 因为m_labels里面初始化为self.n_labels，所以不要动

                if_buffer_i = exist_right + if_beta_i
                # 有没有从i指向buffer的边
                if_i_buffer = exist_left + if_i_beta
                # 有没有从buffer指向i的边
                or_buffer_i = if_buffer_i + if_i_buffer
                final_nr = ~or_buffer_i.unsqueeze(1).expand(
                    -1, len(self.transition_vocab))
                col_nr = torch.zeros(len(self.transition_vocab),
                                     device=words.device).index_fill_(
                                         -1,
                                         torch.tensor(action_id['NR'],
                                                      device=words.device),
                                         1).gt(0)
                nr_mask = final_nr * col_nr
                m_actions.masked_fill_(nr_mask, 1)

                final_lp = ~exist_right
                final_lp_1 = final_lp.unsqueeze(1).expand(
                    -1, len(self.transition_vocab))
                col_lp = torch.zeros(len(self.transition_vocab),
                                     device=words.device).index_fill_(
                                         -1,
                                         torch.tensor(action_id['LP'],
                                                      device=words.device),
                                         1).gt(0)
                lp_mask = final_lp_1 * col_lp
                m_actions.masked_fill_(lp_mask, 1)
                m_labels = m_labels.masked_scatter(lp_mask,
                                                   left_label[final_lp])

                final_rp = ~exist_left
                final_rp_1 = final_rp.unsqueeze(1).expand(
                    -1, len(self.transition_vocab))
                col_rp = torch.zeros(len(self.transition_vocab),
                                     device=words.device).index_fill_(
                                         -1,
                                         torch.tensor(action_id['RP'],
                                                      device=words.device),
                                         1).gt(0)

                rp_mask = final_rp_1 * col_rp
                m_actions.masked_fill_(rp_mask, 1)
                m_labels = m_labels.masked_scatter(rp_mask,
                                                   right_label[final_rp])

                exist_arc = exist_left + exist_right
                final_np = ~exist_arc.unsqueeze(1).expand(
                    -1, len(self.transition_vocab))
                col_np = torch.zeros(len(self.transition_vocab),
                                     device=words.device).index_fill_(
                                         -1,
                                         torch.tensor(action_id['NP'],
                                                      device=words.device),
                                         1).gt(0)
                np_mask = final_np * col_np
                m_actions.masked_fill_(np_mask, 1)

                # 至此m_actions,m_labels填充完毕 [m, len(self.n_transition)]
                # example: m_actions[0]: [0, 0, 1, 1, 0, 1, 0]
                #          m_labels[0] : [self.n_labes, self.n_labes, 4, 2, self.n_labes, 21, self.n_labes]

                back_mask = stack_isnot_clear.unsqueeze(1).expand(
                    -1, self.n_transitions)
                correct_actions = correct_actions.masked_scatter(
                    back_mask, m_actions)
                correct_labels = correct_labels.masked_scatter(
                    back_mask, m_labels)
                # 把stack非空的填回去了

            # 对valid和correct求交集
            # pdb.set_trace()
            valid_mask = valid_actions.gt(0)
            correct_mask = correct_actions.gt(0)
            intersect_mask = valid_mask & correct_mask
            correct_actions = intersect_mask.long()
            correct_mask = correct_actions.gt(0)

            new_correct_labels = torch.tensor(
                [self.n_labels] * self.n_transitions,
                dtype=torch.long,
                device=words.device).unsqueeze(0).expand(k, -1)
            new_correct_labels = new_correct_labels.masked_scatter(
                intersect_mask, correct_labels[intersect_mask])

            windowed = remain_stat[..., :1].clone()
            # pdb.set_trace()
            # [k, 4, 1]
            win_states = windowed[:, 0:3, :]
            null_lstm_mask = win_states.eq(-1)
            win_states = win_states.masked_fill(null_lstm_mask, seq_len)
            # [k, 3, 1]
            s1, s2, s3 = win_states.shape
            win_states = win_states.reshape(s1, s2 * s3).unsqueeze(-1).expand(
                -1, -1, lstm_out_size)
            states_hidden = torch.gather(remain_x, 1,
                                         win_states).reshape(s1, s2, s3, -1)
            # [k, 3, 1, lstm_out_size]
            states_hidden = states_hidden.squeeze(2).reshape(k, -1)
            # [k, 3*lstm_out_size]

            transitions = windowed[:, 3, :]
            # [k, 1, 1]
            transitions_mask = transitions.eq(-1)
            transitions = transitions.masked_fill(transitions_mask,
                                                  self.n_transitions)
            transitions = transitions.reshape(k, -1)
            # [k, 1]
            transitions_embed = self.new_droupout(
                self.transition_embed(transitions)).reshape(k, -1)
            # [k, n_transition_embed]
            final_repr = torch.cat((transitions_embed, states_hidden), -1)
            # [k, n_transition_embed+3*lstm_out_size]

            action_score = self.action_mlp(final_repr)
            # [k, n_transitions]

            # pdb.set_trace()
            new_v = torch.tensor([-float('inf')] * 7,
                                 device=words.device).unsqueeze(0).expand(
                                     k, -1)
            new_v = new_v.masked_scatter(valid_mask, action_score[valid_mask])
            pred_action = torch.argmax(new_v, dim=1)
            # [k] 存放的是预测的action的下标,由于pred_label并不会对state造成啥影响，所以暂时就没算
            new_c = torch.tensor([-float('inf')] * 7,
                                 device=words.device).unsqueeze(0).expand(
                                     k, -1)
            new_c = new_c.masked_scatter(correct_mask,
                                         action_score[correct_mask])
            o_action = torch.argmax(new_c, dim=1)
            # [k] 存放的是oracle action的下标
            o_label = torch.gather(new_correct_labels, 1,
                                   o_action.unsqueeze(1)).squeeze(1)
            # [k] 存放的是oracle label

            # 计算这一步的loss
            # step_action_loss = self.criterion(action_score, o_action)
            step_action_loss = self.criterion(action_score, pred_action)  # 让优化目标就是自己
            this_step_action_embed = self.transition_embed(pred_action)
            # [k, n_transition_embed]
            label_final_repr = torch.cat((final_repr, this_step_action_embed), -1)
            label_score = self.label_mlp(label_final_repr)

            pred_label = torch.argmax(label_score, dim=1)
            step_label_loss = self.criterion(label_score, pred_label)  # 让优化目标就是自己

            # step_label_loss = self.criterion(label_score, o_label)
            step_loss = step_action_loss + step_label_loss
            # 已经对batch求了平均
            losses.append(step_loss)
            # 现在这边是把每一步的loss求和后平均再反向传播，如果这造成现存爆炸，可以一步之后就backward（还是一个batch再update）
            

            pro = torch.rand(k, device=words.device)
            # [k] 对应k个状态follow pred的概率,小于p则follow pred的，大于则follow o_action
            follow_pred = pro.lt(p)
            followed_action = pred_action.masked_scatter(~follow_pred, o_action[~follow_pred])
            # [k] k个state真正要执行的action的下标

            # 之后就是要对这k个state去进行操作了
            # TODO: 这种写法不好，要确保只是只包含一个
            Reduce_mask = followed_action.eq(action_id['LR'][0]) + followed_action.eq(action_id['NR'][0])
            Pass_mask = followed_action.eq(action_id['LP'][0]) + followed_action.eq(action_id['NP'][0]) + followed_action.eq(action_id['RP'][0])
            Shift_mask = followed_action.eq(action_id['RS'][0]) + followed_action.eq(action_id['NS'][0])
            # [k]

            with torch.no_grad():
                require_reduce = remain_stat[Reduce_mask]
                require_reduce[:, 0, :-1] = require_reduce[:, 0, 1:].clone()
                require_reduce[:, 0, -1] = -1
                reduce_action_mask = ~require_reduce.ge(-1)
                reduce_action_mask[:, 3, 0] = True
                require_reduce = require_reduce.masked_scatter(reduce_action_mask, followed_action[Reduce_mask])
                back_reduce_mask = Reduce_mask.unsqueeze(1).expand(-1, 4).unsqueeze(2).expand(-1, -1, seq_len)
                remain_stat = remain_stat.masked_scatter(back_reduce_mask, require_reduce)

                require_pass = remain_stat[Pass_mask]
                require_pass[:, 2, 1:] = require_pass[:, 2, :-1].clone()
                require_pass[:, 2, 0] = require_pass[:, 0, 0]
                require_pass[:, 0, :-1] = require_pass[:, 0, 1:].clone()
                require_pass[:, 0, -1] = -1
                pass_action_mask = ~require_pass.ge(-1)
                pass_action_mask[:, 3, 0] = True
                require_pass = require_pass.masked_scatter(pass_action_mask, followed_action[Pass_mask])
                back_pass_mask = Pass_mask.unsqueeze(1).expand(-1, 4).unsqueeze(2).expand(-1, -1, seq_len)
                remain_stat = remain_stat.masked_scatter(back_pass_mask, require_pass)

                require_shift = remain_stat[Shift_mask]
                buffer_top = require_shift[:, 1, 0].unsqueeze(1)
                reverse_deque = torch.flip(require_shift[:, 2], [1])
                b_rd = torch.cat((buffer_top, reverse_deque), -1)
                b_rd_mask = b_rd.gt(-1)
                cat_stack = require_shift[:, 0]
                cat_mask = cat_stack.ge(-1)
                b_rd_s = torch.cat((b_rd, cat_stack), -1)
                f_mask = torch.cat((b_rd_mask, cat_mask), -1)
                stack_mask = ~require_shift.ge(-1)
                stack_mask[:, 0] = True
                require_shift = require_shift.masked_scatter(stack_mask, b_rd_s[f_mask.cumsum(1).le(seq_len)&f_mask])
                require_shift[:, 1, :-1] = require_shift[:, 1, 1:].clone()
                require_shift[:, 1, -1] = -1
                require_shift[:, 2] = -1
                shift_action_mask = ~require_shift.ge(-1)
                shift_action_mask[:, 3, 0] = True
                require_shift = require_shift.masked_scatter(shift_action_mask, followed_action[Shift_mask])
                back_shift_mask = Shift_mask.unsqueeze(1).expand(-1, 4).unsqueeze(2).expand(-1, -1, seq_len)
                remain_stat = remain_stat.masked_scatter(back_shift_mask, require_shift)
            # 到此remain_stat修改完毕,接下来把状态写回save_stat
            state_back_mask = unfinish_mask.unsqueeze(1).expand(-1, 4).unsqueeze(2).expand(-1, -1, seq_len)
            save_stat = save_stat.masked_scatter(state_back_mask, remain_stat)
            unfinish_mask = save_stat[:, 1, 0].gt(-1)
            # [batch_size]
        
        # losses中存储了每一步的loss
        # loss = torch.sum(torch.stack(losses)) / (len(losses) + 1)
        # print(f'\nsteps:{step}')
        losses = torch.stack(losses) * torch.tensor(remain_ks, device=words.device)
        loss = torch.sum(losses) / (len(losses) + 1)
        return loss, remain_ks

    def batch_decode(self, words, words_len, feats):
        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)

        feat_embeds = []
        if 'tag' in self.args.feat:
            feat_embeds.append(self.tag_embed(feats.pop()))
        if 'char' in self.args.feat:
            feat_embeds.append(self.char_embed(feats.pop(0)))
        if 'bert' in self.args.feat:
            feat_embeds.append(self.bert_embed(feats.pop(0)))
        if 'lemma' in self.args.feat:
            feat_embeds.append(self.lemma_embed(feats.pop(0)))
        word_embed, feat_embed = self.embed_dropout(word_embed,
                                                    torch.cat(feat_embeds, -1))
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        x = pack_padded_sequence(embed, mask.sum(1), True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)
        # x:[batch_size, seq_len, hidden_size*2]
        null_hidden = self.null_lstm_hidden.expand(batch_size, -1, -1)
        x = torch.cat([x, null_hidden], dim=1)
        # x:[batch_size, seq_len+1, hidden_size*2]
        lstm_out_size = x.shape[2]

        sdt = torch.tensor([[0] + [-1] *
                            (seq_len - 1), [-1] * seq_len, [-1] * seq_len],
                           dtype=torch.long,
                           device=words.device)
        # 这边先用-1pad，之后取的时候，要先fill，stack,buffer,deque用seq_len,pre_action用n_transitions
        sdt = sdt.unsqueeze(0).expand(batch_size, -1, -1)
        bu_lst = []
        for length in words_len:
            bu_lst.append(
                torch.arange(1,
                             length.item(),
                             dtype=torch.long,
                             device=words.device))
        bu_out = pad(bu_lst, -1,
                     seq_len).to(words.device).reshape(batch_size, 1, -1)
        init_stat = torch.cat((sdt, bu_out), dim=1)
        save_stat = init_stat[:, [0, 3, 2, 1]]
        # 初始状态:[batch _size, 4, seq_len] stack,buffer,deque,pre_action
        # pad with -1
        # remain_edge = edges
        # remain_label = labels
        # remain_x = x
        unfinish_mask = save_stat[:, 1, 0].gt(-1)

        action_id = {
            action_: [
                self.transition_vocab.stoi[a]
                for a in self.transition_vocab.stoi.keys()
                if a.startswith(action_)
            ]
            for action_ in ["LR", "LP", "RS", "RP", "NS", "NR", "NP"]
        }
        
        edges = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.long, device=words.device)
        labels = -torch.ones((batch_size, seq_len, seq_len), dtype=torch.long, device=words.device)

        step = 0
        while (unfinish_mask.sum()):
            step += 1
            remain_stat = save_stat[unfinish_mask]
            # [k, 4, seq_len]
            remain_edge = edges[unfinish_mask]
            remain_label = labels[unfinish_mask]
            # remain_stat, remain_edge, remain_label都要写回
            # [k, seq_len, seq_len]
            remain_x = x[unfinish_mask]
            # [k, seq_len+1, lstm_out_size]
            k = remain_stat.shape[0]

            valid1 = torch.ones((k, len(self.transition_vocab)),
                                dtype=torch.long,
                                device=words.device)
            base_mask = valid1.gt(0)
            valid1_mask = base_mask.clone()
            stat_len = remain_stat.gt(-1).sum(dim=2)
            # [k, 4]
            valid_mask_sg1 = stat_len[:, 0].gt(1)
            # [k]
            valid_mask_bg0 = stat_len[:, 1].gt(0)
            # [k]
            valid_mask_sg0 = stat_len[:, 0].gt(0)
            # [k]
            mask_v1 = valid_mask_sg1 & valid_mask_bg0
            mask_v1 = mask_v1.unsqueeze(1)
            valid1_mask = valid1_mask & mask_v1
            col_cond1 = torch.zeros(
                len(self.transition_vocab), device=words.device).index_fill_(
                    -1,
                    torch.tensor(action_id['LR'] + action_id['LP'] +
                                 action_id['RP'],
                                 device=words.device), 1).gt(0)
            valid1_mask = valid1_mask * col_cond1
            valid1.masked_fill_(~valid1_mask, 0)

            valid2 = torch.ones((k, len(self.transition_vocab)),
                                dtype=torch.long,
                                device=words.device)
            valid2_mask = base_mask.clone()

            mask_v2 = valid_mask_bg0.unsqueeze(1)
            valid2_mask = valid2_mask & mask_v2
            col_cond2 = torch.zeros(len(self.transition_vocab),
                                    device=words.device).index_fill_(
                                        -1,
                                        torch.tensor(action_id['NS'],
                                                     device=words.device),
                                        1).gt(0)
            valid2_mask = valid2_mask * col_cond2
            valid2.masked_fill_(~valid2_mask, 0)

            valid3 = torch.ones((k, len(self.transition_vocab)),
                                dtype=torch.long,
                                device=words.device)
            valid3_mask = base_mask.clone()
            mask_v3 = valid_mask_bg0 & valid_mask_sg0
            mask_v3 = mask_v3.unsqueeze(1)
            valid3_mask = valid3_mask & mask_v3
            col_cond3 = torch.zeros(len(self.transition_vocab),
                                    device=words.device).index_fill_(
                                        -1,
                                        torch.tensor(action_id['RS'],
                                                     device=words.device),
                                        1).gt(0)
            valid3_mask = valid3_mask * col_cond3
            valid3.masked_fill_(~valid3_mask, 0)

            valid4 = torch.ones((k, len(self.transition_vocab)),
                                dtype=torch.long,
                                device=words.device)
            valid4_mask = base_mask.clone()
            msak_v4 = valid_mask_sg1.unsqueeze(1)
            valid4_mask = valid4_mask & msak_v4
            col_cond4 = torch.zeros(len(self.transition_vocab),
                                    device=words.device).index_fill_(
                                        -1,
                                        torch.tensor(
                                            action_id['NR'] + action_id['NP'],
                                            device=words.device), 1).gt(0)
            valid4_mask = valid4_mask * col_cond4
            valid4.masked_fill_(~valid4_mask, 0)
            valid_actions = valid1 + valid2 + valid3 + valid4
            # [k, 7]
            valid_mask = valid_actions.gt(0)

            windowed = remain_stat[..., :1].clone()
            # pdb.set_trace()
            # [k, 4, 1]
            win_states = windowed[:, 0:3, :]
            null_lstm_mask = win_states.eq(-1)
            win_states = win_states.masked_fill(null_lstm_mask, seq_len)
            # [k, 3, 1]
            s1, s2, s3 = win_states.shape
            win_states = win_states.reshape(s1, s2 * s3).unsqueeze(-1).expand(
                -1, -1, lstm_out_size)
            states_hidden = torch.gather(remain_x, 1,
                                         win_states).reshape(s1, s2, s3, -1)
            # [k, 3, 1, lstm_out_size]
            states_hidden = states_hidden.squeeze(2).reshape(k, -1)
            # [k, 3*lstm_out_size]

            transitions = windowed[:, 3, :]
            # [k, 1, 1]
            transitions_mask = transitions.eq(-1)
            transitions = transitions.masked_fill(transitions_mask,
                                                  self.n_transitions)
            transitions = transitions.reshape(k, -1)
            # [k, 1]
            transitions_embed = self.new_droupout(
                self.transition_embed(transitions)).reshape(k, -1)
            # [k, n_transition_embed]
            final_repr = torch.cat((transitions_embed, states_hidden), -1)
            # [k, n_transition_embed+3*lstm_out_size]

            action_score = self.action_mlp(final_repr)
            # [k, n_transitions]
            new_v = torch.tensor([-float('inf')] * 7,
                                 device=words.device).unsqueeze(0).expand(
                                     k, -1)
            new_v = new_v.masked_scatter(valid_mask, action_score[valid_mask])
            pred_action = torch.argmax(new_v, dim=1)
            # [k] 存放的是预测的action的下标
            this_step_action_embed = self.transition_embed(pred_action)
            # [k, n_transition_embed]
            label_final_repr = torch.cat((final_repr, this_step_action_embed), -1)
            label_score = self.label_mlp(label_final_repr)
            pred_label = torch.argmax(label_score, dim=1)
            require_left_mask = pred_action.eq(action_id['LR'][0]) + pred_action.eq(action_id['LP'][0])
            require_right_mask = pred_action.eq(action_id['RP'][0]) + pred_action.eq(action_id['RS'][0])

            # 加边
            # pdb.set_trace()
            with torch.no_grad():
                require_left = remain_stat[require_left_mask]
                req_left_edge = remain_edge[require_left_mask]
                req_left_label = remain_label[require_left_mask]
                t = require_left.shape[0]
                x_h = require_left[:, 0, 0].reshape(t, 1, 1).expand(-1, -1, seq_len)
                y_h = require_left[:, 1, 0].reshape(t, 1, 1).expand(-1, seq_len, -1)
                x_m = (~req_left_edge.ge(0)).scatter(1, x_h, 1)
                y_m = (~req_left_edge.ge(0)).scatter(2, y_h, 1)
                final_m = x_m & y_m
                req_left_edge = req_left_edge.masked_fill(final_m, 1)
                req_left_label = req_left_label.masked_scatter(final_m, pred_label[require_left_mask])
                left_back_mask = require_left_mask.unsqueeze(1).expand(-1, seq_len).unsqueeze(2).expand(-1,-1,seq_len)
                remain_edge = remain_edge.masked_scatter(left_back_mask, req_left_edge)
                remain_label = remain_label.masked_scatter(left_back_mask, req_left_label)

                require_right = remain_stat[require_right_mask]
                req_right_edge = remain_edge[require_right_mask]
                req_right_label = remain_label[require_right_mask]
                t = require_right.shape[0]
                x_h = require_right[:, 1, 0].reshape(t, 1, 1).expand(-1, -1, seq_len)
                y_h = require_right[:, 0, 0].reshape(t, 1, 1).expand(-1, seq_len, -1)
                x_m = (~req_right_edge.ge(0)).scatter(1, x_h, 1)
                y_m = (~req_right_edge.ge(0)).scatter(2, y_h, 1)
                final_m = x_m & y_m
                req_right_edge = req_right_edge.masked_fill(final_m, 1)
                req_right_label = req_right_label.masked_scatter(final_m, pred_label[require_right_mask])
                right_back_mask = require_right_mask.unsqueeze(1).expand(-1, seq_len).unsqueeze(2).expand(-1,-1,seq_len)
                remain_edge = remain_edge.masked_scatter(right_back_mask, req_right_edge)
                remain_label = remain_label.masked_scatter(right_back_mask, req_right_label)
                # 最后返回label的时候要把self.n_labels改成-1

                # 把edge和label回写
                edge_back_mask = unfinish_mask.unsqueeze(1).expand(-1, seq_len).unsqueeze(2).expand(-1, -1, seq_len)
                edges = edges.masked_scatter(edge_back_mask, remain_edge)
                labels = labels.masked_scatter(edge_back_mask, remain_label)
            # pdb.set_trace()
            
            followed_action = pred_action
            Reduce_mask = followed_action.eq(action_id['LR'][0]) + followed_action.eq(action_id['NR'][0])
            Pass_mask = followed_action.eq(action_id['LP'][0]) + followed_action.eq(action_id['NP'][0]) + followed_action.eq(action_id['RP'][0])
            Shift_mask = followed_action.eq(action_id['RS'][0]) + followed_action.eq(action_id['NS'][0])
            # [k]

            with torch.no_grad():
                require_reduce = remain_stat[Reduce_mask]
                require_reduce[:, 0, :-1] = require_reduce[:, 0, 1:].clone()
                require_reduce[:, 0, -1] = -1
                reduce_action_mask = ~require_reduce.ge(-1)
                reduce_action_mask[:, 3, 0] = True
                require_reduce = require_reduce.masked_scatter(reduce_action_mask, followed_action[Reduce_mask])
                back_reduce_mask = Reduce_mask.unsqueeze(1).expand(-1, 4).unsqueeze(2).expand(-1, -1, seq_len)
                remain_stat = remain_stat.masked_scatter(back_reduce_mask, require_reduce)

                require_pass = remain_stat[Pass_mask]
                require_pass[:, 2, 1:] = require_pass[:, 2, :-1].clone()
                require_pass[:, 2, 0] = require_pass[:, 0, 0]
                require_pass[:, 0, :-1] = require_pass[:, 0, 1:].clone()
                require_pass[:, 0, -1] = -1
                pass_action_mask = ~require_pass.ge(-1)
                pass_action_mask[:, 3, 0] = True
                require_pass = require_pass.masked_scatter(pass_action_mask, followed_action[Pass_mask])
                back_pass_mask = Pass_mask.unsqueeze(1).expand(-1, 4).unsqueeze(2).expand(-1, -1, seq_len)
                remain_stat = remain_stat.masked_scatter(back_pass_mask, require_pass)

                require_shift = remain_stat[Shift_mask]
                buffer_top = require_shift[:, 1, 0].unsqueeze(1)
                reverse_deque = torch.flip(require_shift[:, 2], [1])
                b_rd = torch.cat((buffer_top, reverse_deque), -1)
                b_rd_mask = b_rd.gt(-1)
                cat_stack = require_shift[:, 0]
                cat_mask = cat_stack.ge(-1)
                b_rd_s = torch.cat((b_rd, cat_stack), -1)
                f_mask = torch.cat((b_rd_mask, cat_mask), -1)
                stack_mask = ~require_shift.ge(-1)
                stack_mask[:, 0] = True
                require_shift = require_shift.masked_scatter(stack_mask, b_rd_s[f_mask.cumsum(1).le(seq_len)&f_mask])
                require_shift[:, 1, :-1] = require_shift[:, 1, 1:].clone()
                require_shift[:, 1, -1] = -1
                require_shift[:, 2] = -1
                shift_action_mask = ~require_shift.ge(-1)
                shift_action_mask[:, 3, 0] = True
                require_shift = require_shift.masked_scatter(shift_action_mask, followed_action[Shift_mask])
                back_shift_mask = Shift_mask.unsqueeze(1).expand(-1, 4).unsqueeze(2).expand(-1, -1, seq_len)
                remain_stat = remain_stat.masked_scatter(back_shift_mask, require_shift)
            # 到此remain_stat修改完毕,接下来把状态写回save_stat
            state_back_mask = unfinish_mask.unsqueeze(1).expand(-1, 4).unsqueeze(2).expand(-1, -1, seq_len)
            save_stat = save_stat.masked_scatter(state_back_mask, remain_stat)
            unfinish_mask = save_stat[:, 1, 0].gt(-1)
            # [batch_size]

        labels = labels.masked_fill(labels.eq(self.n_labels), -1)
        return edges, labels
        

    def new_decode(self, words, words_len, feats):
        r"""
        words: [batch_size, seq_len]
        words_len: [batch_size]
        第二阶段不依赖第一阶段 可能对用dynamic oracle训练更好
        """
        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)

        feat_embeds = []
        if 'tag' in self.args.feat:
            feat_embeds.append(self.tag_embed(feats.pop()))
        if 'char' in self.args.feat:
            feat_embeds.append(self.char_embed(feats.pop(0)))
        if 'bert' in self.args.feat:
            feat_embeds.append(self.bert_embed(feats.pop(0)))
        if 'lemma' in self.args.feat:
            feat_embeds.append(self.lemma_embed(feats.pop(0)))
        word_embed, feat_embed = word_embed, torch.cat(feat_embeds, -1)
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        x = pack_padded_sequence(embed, mask.sum(1), True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        # x:[batch_size, seq_len, hidden_size*2]
        null_hidden = self.null_lstm_hidden.expand(batch_size, -1, -1)
        x = torch.cat([x, null_hidden], dim=1)
        # x:[batch_size, seq_len+1, hidden_size*2]
        lstm_out_size = x.shape[2]

        arcs = []
        labels = []

        action_id = {
            action_: [
                self.transition_vocab.stoi[a]
                for a in self.transition_vocab.stoi.keys()
                if a.startswith(action_)
            ]
            for action_ in ["LR", "LP", "RS", "RP", "NS", "NR", "NP"]
        }

        for i in range(batch_size):
            look_up = x[i]
            arc = [[0] * seq_len for k in range(seq_len)]
            label = [[-1] * seq_len for k in range(seq_len)]

            actions = [self.n_transitions] * self.window  # 动作序列,-1头
            stack = [seq_len] * self.window + [0]  # 正在处理的节点序列,-1头
            buffer = [k for k in range(1, words_len[i].item())
                      ] + [seq_len] * self.window  # 待处理的节点序列,0头
            deque = [seq_len] * self.window  # 暂时跳过的节点序列（可能存在多个头节点）,-1头
            buffer_len = words_len[i].item() - 1
            stack_len = 1
            count = 0

            h_t = torch.zeros(
                1, 1,
                (self.args.n_lstm_hidden * 6 + self.args.n_transition_embed) //
                3).to(x.device)
            c_t = torch.zeros(
                1, 1,
                (self.args.n_lstm_hidden * 6 + self.args.n_transition_embed) //
                3).to(x.device)

            while (buffer_len > 0):
                if (count > 500):
                    print('count supass 500')
                    break
                count += 1

                valid_actions = []

                if (stack_len > 1 and buffer_len > 0):
                    valid_actions += action_id['LR']
                    valid_actions += action_id['LP']
                    valid_actions += action_id['RP']

                if (buffer_len > 0):
                    valid_actions += action_id['NS']
                    if (stack_len > 0):
                        valid_actions += action_id['RS']  # ROOT,NULL

                if (stack_len > 1):
                    valid_actions += action_id['NR']
                    valid_actions += action_id['NP']

                action = valid_actions[0]
                final_repr = None
                if len(valid_actions) > 1:
                    stack_embed = torch.sum(look_up[stack[-self.window:]],
                                            dim=0)
                    buffer_embed = torch.sum(look_up[buffer[0:self.window]],
                                             dim=0)
                    deque_embed = torch.sum(look_up[deque[-self.window:]],
                                            dim=0)
                    transition_embed = torch.sum(self.transition_embed(
                        torch.tensor(actions[-self.window:],
                                     dtype=torch.long,
                                     device=x.device)),
                                                 dim=0)
                    final_repr = torch.cat((transition_embed, stack_embed,
                                            buffer_embed, deque_embed),
                                           dim=-1).unsqueeze(0)

                    if (self.decode_mode == 'mlp'):
                        score = self.base_mlp(final_repr).squeeze(0)[
                            torch.tensor(valid_actions,
                                         dtype=torch.long,
                                         device=x.device)]
                    elif (self.decode_mode == 'att'):

                        # [1,n_transition_embed+3*lstm_out_size]
                        states_repr = self.state_mlp(final_repr)
                        # [1, n_mlp]
                        transitions_repr = self.transition_embed(
                            torch.tensor(
                                [i for i in range(self.n_transitions)],
                                device=states_repr.device))
                        transitions_final = self.action_mlp(
                            transitions_repr).transpose(1, 0)
                        # [n_mlp, n_transitions]
                        score = torch.matmul(
                            states_repr,
                            transitions_final).squeeze(0)[torch.tensor(
                                valid_actions,
                                dtype=torch.long,
                                device=x.device)]

                    elif (self.decode_mode == 'lstm'):

                        # [1,n_transition_embed+3*lstm_out_size]
                        final_repr = final_repr.unsqueeze(0)
                        out, (h_t,
                              c_t) = self.decoder_lstm(final_repr, (h_t, c_t))
                        out = out.squeeze(0)
                        score = self.decoder_lstm_mlp(out).squeeze(0)[
                            torch.tensor(valid_actions,
                                         dtype=torch.long,
                                         device=x.device)]

                    elif (self.decode_mode == 'beta'):
                        plus = torch.stack(
                            (stack_embed, buffer_embed, deque_embed))
                        inter = self.inter.reshape(3, 1).expand(
                            -1, plus.shape[1])
                        plus = plus * inter
                        plus = torch.sum(plus, dim=0)
                        # [lstm_hiddem*2]
                        final_repr = torch.cat((transition_embed, plus),
                                               -1).unsqueeze(0)
                        score = self.mlp(final_repr).squeeze(0)[torch.tensor(
                            valid_actions, dtype=torch.long, device=x.device)]

                    elif (self.decode_mode == 'dual'):
                        score = self.action_mlp(final_repr).squeeze(0)[
                            torch.tensor(valid_actions,
                                         dtype=torch.long,
                                         device=x.device)]

                    action_idx = torch.argmax(score, dim=0).item()
                    action = valid_actions[action_idx]

                #  这个第二阶段就是不依赖于第一阶段的
                if action in action_id["LR"] or action in action_id["LP"] or \
                        action in action_id["RS"] or action in action_id["RP"]:
                    if action in action_id["RS"] or action in action_id["RP"]:
                        head = stack[-1]
                        modifier = buffer[0]
                    else:
                        head = buffer[0]
                        modifier = stack[-1]

                    arc[modifier][head] = 1
                    if (self.decode_mode != 'dual'):
                        label[modifier][head] = int(
                            self.transition_vocab.itos[action].split(':')[1])
                    else:
                        if (final_repr is None):
                            stack_embed = torch.sum(
                                look_up[stack[-self.window:]], dim=0)
                            buffer_embed = torch.sum(
                                look_up[buffer[0:self.window]], dim=0)
                            deque_embed = torch.sum(
                                look_up[deque[-self.window:]], dim=0)
                            transition_embed = torch.sum(self.transition_embed(
                                torch.tensor(actions[-self.window:],
                                             dtype=torch.long,
                                             device=x.device)),
                                                         dim=0)
                            final_repr = torch.cat(
                                (transition_embed, stack_embed, buffer_embed,
                                 deque_embed),
                                dim=-1).unsqueeze(0)

                        current_action_embed = self.transition_embed(
                            torch.tensor([action],
                                         dtype=torch.long,
                                         device=x.device))
                        label_final = torch.cat(
                            (final_repr, current_action_embed), -1)
                        label_score = self.label_mlp(label_final).squeeze(0)[
                            torch.tensor([vla for vla in range(self.n_labels)],
                                         dtype=torch.long,
                                         device=x.device)]
                        label_idx = torch.argmax(label_score, dim=0).item()
                        label[modifier][head] = label_idx

                actions.append(action)

                # reduce
                if action in action_id["LR"] or action in action_id["NR"]:
                    stack.pop(-1)
                    stack_len -= 1

                # pass
                elif action in action_id["LP"] or action in action_id[
                        "NP"] or action in action_id["RP"]:
                    stack_top = stack.pop(-1)
                    stack_len -= 1
                    deque.append(stack_top)

                # shift
                elif action in action_id["RS"] or action in action_id["NS"]:
                    j = len(deque) - 1
                    start = j
                    while (deque[j] != seq_len):
                        stack.append(deque.pop(-1))
                        j -= 1
                    stack_len += start - j

                    buffer_top = buffer.pop(0)
                    stack.append(buffer_top)
                    stack_len += 1
                    buffer_len -= 1

            arcs.append(arc)
            labels.append(label)

        return torch.tensor(arcs, device=words.device), torch.tensor(
            labels, device=words.device)