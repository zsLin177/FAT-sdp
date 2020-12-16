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

    def dynamic_loss3(self, words, words_len, feats, edges, labels):
        r"""
        目前只适用dual
        目前的优化目标只是Correct(c)中分数最高的
        words: [batch_size, seq_len]
        words_len: [batch_size]
        edges : [batch_size, seq_len, seq_len]
        labels: [batch_size, seq_len, seq_len]
        对图中没有的边就不加到correct中（有可能会造成correct为空)
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
        sdt = sdt.unsqueeze(0).expand(batch_size, -1, -1)
        bu_lst = []
        for length in words_len:
            bu_lst.append(
                torch.arange(1,
                             length.item(),
                             dtype=torch.long,
                             device=words.device))
        bu_out = pad(bu_lst, -1, seq_len).to(words.device).reshape(batch_size, 1, -1)
        remain_stat = torch.cat((sdt, bu_out), dim=1)
        remain_stat = remain_stat[:, [0, 3, 2, 1]]
        # 初始状态:[batch_size, 4, seq_len] stack,buffer,deque,pre_action
        # pad with -1
        finish_mask = remain_stat[:, 1, 0].gt(-1)

        while(finish_mask.sum()):
            remain_stat = remain_stat[finish_mask]
            # [k, 4, seq_len]
            








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