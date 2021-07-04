# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from supar.models import TransitionSemanticDependencyModel
from supar.parsers.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import bos, pad, unk
from supar.utils.field import ChartField, Field, SubwordField, TransitionField, StateField, ParStateField, ParTransitionField, TranLabelField
from supar.utils.logging import get_logger, progress_bar, init_logger, logger
from supar.utils.metric import ChartMetric
from supar.utils.transform import CoNLL
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from supar.utils.metric import Metric
from supar.utils.parallel import DistributedDataParallel as DDP
from supar.utils.parallel import is_master
import pdb
import random
import math

# logger = get_logger(__name__)


class TransitionSemanticDependencyParser(Parser):
    r"""
    The implementation of Transition-based Semantic Dependency Parser.
    """

    NAME = 'transition_based-semantic-dependency'
    MODEL = TransitionSemanticDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = self.args.update(locals())
        self.WORD, self.CHAR, self.BERT = self.transform.FORM
        self.LEMMA = self.transform.LEMMA
        self.TAG = self.transform.POS
        if (args.decode_mode in ('dual', 'par-dual')):
            self.EDGE, self.LABEL, self.Transition, self.Translabel, self.State = self.transform.PHEAD  # for decode_mode = dual
        else:
            self.EDGE, self.LABEL, self.Transition, self.State = self.transform.PHEAD

    def dynamic_train(self,
                      train,
                      dev,
                      test,
                      buckets=32,
                      batch_size=5000,
                      verbose=True,
                      clip=5.0,
                      epochs=5000,
                      patience=150,
                      **kwargs):

        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        p = args.pro
        k = args.k

        mu = args.mu
        if (mu > 0):
            p = 0.0
            logger.info(f"pro start from {p}")

        random.seed(args.seed)
        self.transform.train()
        if dist.is_initialized():
            args.batch_size = args.batch_size // dist.get_world_size()
        logger.info("Loading the data")
        train = Dataset(self.transform, args.train, **args)
        dev = Dataset(self.transform, args.dev)
        test = Dataset(self.transform, args.test)
        train.build(args.batch_size, args.buckets, True, dist.is_initialized())
        # pdb.set_trace()
        dev.build(args.batch_size, args.buckets)
        test.build(args.batch_size, args.buckets)
        logger.info(
            f"\n{'train:':6} {train}\n{'dev:':6} {dev}\n{'test:':6} {test}\n")

        if dist.is_initialized():
            self.model = DDP(self.model,
                             device_ids=[args.local_rank],
                             find_unused_parameters=True)

        elapsed = timedelta()
        best_e, best_metric = 1, Metric()

        for epoch in range(1, args.epochs + 1):
            start = datetime.now()

            logger.info(f"Epoch {epoch} / {args.epochs}:")
            if (epoch <= k):
                self._train(train.loader, args.decode_mode)
            else:
                if (args.batch_train):
                    logger.info(f"dynamic pro:{p}")
                    # ratio = self._dynamic_train(train.loader, args.decode_mode,
                    #                             p)
                    self._dynamic_train(train.loader, args.decode_mode, p)
                    # logger.info(f"epoch follow false ratio:{ratio}")
                    if (mu > 0):
                        p = 1.0 - (mu / (mu + math.exp(epoch / mu)))
                else:
                    self._nobatch_dynamic_train(train.loader, args.decode_mode,
                                                p)
            dev_metric = self._batch_evaluate(dev.loader, args.decode_mode)
            logger.info(f"{'dev:':6} - {dev_metric}")
            test_metric = self._batch_evaluate(test.loader, args.decode_mode)
            logger.info(f"{'test:':6} - {test_metric}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                if is_master():
                    self.save(args.path)
                logger.info(f"{t}s elapsed (saved)\n")
            else:
                logger.info(f"{t}s elapsed\n")
            elapsed += t
            if epoch - best_e >= args.patience:
                break
        loss, metric = self.load(**args)._batch_evaluate(test.loader)

        logger.info(f"Epoch {best_e} saved")
        logger.info(f"{'dev:':6} {best_metric}")
        logger.info(f"{'test:':6} {metric}")
        logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")

    def train(self,
              train,
              dev,
              test,
              buckets=32,
              batch_size=5000,
              verbose=True,
              clip=5.0,
              epochs=5000,
              patience=100,
              **kwargs):
        r"""
        Args:
            train/dev/test (list[list] or str):
                Filenames of the train/dev/test datasets.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for training.
        """
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        if dist.is_initialized():
            args.batch_size = args.batch_size // dist.get_world_size()
        logger.info("Loading the data")
        train = Dataset(self.transform, args.train, **args)
        dev = Dataset(self.transform, args.dev)
        test = Dataset(self.transform, args.test)
        train.build(args.batch_size, args.buckets, True, dist.is_initialized())
        dev.build(args.batch_size, args.buckets)
        test.build(args.batch_size, args.buckets)
        logger.info(
            f"\n{'train:':6} {train}\n{'dev:':6} {dev}\n{'test:':6} {test}\n")

        if dist.is_initialized():
            self.model = DDP(self.model,
                             device_ids=[args.local_rank],
                             find_unused_parameters=True)

        elapsed = timedelta()
        best_e, best_metric = 1, Metric()

        for epoch in range(1, args.epochs + 1):
            start = datetime.now()

            logger.info(f"Epoch {epoch} / {args.epochs}:")
            self._train(train.loader, args.decode_mode)
            dev_metric = self._batch_evaluate(dev.loader, args.decode_mode)
            logger.info(f"{'dev:':6} - {dev_metric}")
            test_metric = self._batch_evaluate(test.loader, args.decode_mode)
            logger.info(f"{'test:':6} - {test_metric}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                if is_master():
                    self.save(args.path)
                logger.info(f"{t}s elapsed (saved)\n")
            else:
                logger.info(f"{t}s elapsed\n")
            elapsed += t
            if epoch - best_e >= args.patience:
                break
        loss, metric = self.load(**args)._batch_evaluate(test.loader)

        logger.info(f"Epoch {best_e} saved")
        logger.info(f"{'dev:':6} {best_metric}")
        logger.info(f"{'test:':6} {metric}")
        logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")

    def evaluate(self,
                 data,
                 buckets=8,
                 batch_size=5000,
                 verbose=True,
                 **kwargs):
        r"""
        Args:
            data (str):
                The data for evaluation, both list of instances and filename are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for evaluation.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self,
                data,
                pred=None,
                buckets=8,
                batch_size=5000,
                verbose=True,
                **kwargs):
        r"""
        Args:
            data (list[list] or str):
                The data for prediction, both a list of instances and filename are allowed.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            prob (bool):
                If ``True``, outputs the probabilities. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for prediction.

        Returns:
            A :class:`~supar.utils.Dataset` object that stores the predicted results.
        """

        return super().predict(**Config().update(locals()))

    def _nobatch_dynamic_train(self, loader, mode, p):
        # 目前仅针对了dual loss
        self.model.train()
        bar = progress_bar(loader)
        for words, *feats, edges, labels, transitions, translabel, states in bar:
            pro = random.random()
            if (pro > p):
                self.optimizer.zero_grad()
                # mask = words.ne(self.WORD.pad_index)
                # mask = mask.unsqueeze(1) & mask.unsqueeze(2)
                # mask[:, 0] = 0
                transition_mask = transitions.ge(0)
                transition_len = transition_mask.sum(1)
                # pdb.set_trace()
                action_score, label_score = self.model(words, feats, states,
                                                       transitions,
                                                       transition_len)
                loss = self.model.dual_loss(action_score, label_score,
                                            transitions, translabel)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.args.clip)
                self.optimizer.step()
                self.scheduler.step()

                # edge_preds, label_preds = self.model.decode(s_edge, s_label)
                # metric(label_preds.masked_fill(~(edge_preds.gt(0) & mask), -1),
                #        labels.masked_fill(~(edges.gt(0) & mask), -1))
                bar.set_postfix_str(
                    f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}"
                )
            else:
                self.optimizer.zero_grad()

                # transition_mask = transitions.ge(0)
                # transition_len = transition_mask.sum(1)
                mask = words.ne(self.WORD.pad_index)
                words_len = torch.sum(mask, dim=-1)

                # loss = self.model.dynamic_loss4(words, words_len, feats, edges,
                #                                 labels, p)

                # 6,7是分析实验
                loss = self.model.dynamic_loss2(words, words_len, feats, edges,
                                                labels)
                # all_pred_step += num_follow_pred_step
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.args.clip)
                self.optimizer.step()
                self.scheduler.step()

                # edge_preds, label_preds = self.model.decode(s_edge, s_label)
                # metric(label_preds.masked_fill(~(edge_preds.gt(0) & mask), -1),
                #        labels.masked_fill(~(edges.gt(0) & mask), -1))
                bar.set_postfix_str(
                    f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}"
                )

    def _dynamic_train(self, loader, mode, p):
        # num_all = 0
        # num_false = 0
        self.model.train()
        bar = progress_bar(loader)
        if (mode not in ('dual', 'par-dual')):
            for words, *feats, edges, labels, transitions, states in bar:
                self.optimizer.zero_grad()

                # transition_mask = transitions.ge(0)
                # transition_len = transition_mask.sum(1)
                mask = words.ne(self.WORD.pad_index)
                words_len = torch.sum(mask, dim=-1)

                # loss = self.model.dynamic_loss4(words, words_len, feats, edges,
                #                                 labels, p)

                loss = self.model.dynamic_loss4_single_fix(words, words_len, feats,
                                                       edges, labels, p)
                # num_all += n_all
                # num_false += n_false
                # all_pred_step += num_follow_pred_step
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.args.clip)
                self.optimizer.step()
                self.scheduler.step()

                # edge_preds, label_preds = self.model.decode(s_edge, s_label)
                # metric(label_preds.masked_fill(~(edge_preds.gt(0) & mask), -1),
                #        labels.masked_fill(~(edges.gt(0) & mask), -1))
                bar.set_postfix_str(
                    f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}"
                )
            # return num_false / num_all
        else:
            for words, *feats, edges, labels, transitions, translabel, states in bar:
                self.optimizer.zero_grad()

                # transition_mask = transitions.ge(0)
                # transition_len = transition_mask.sum(1)
                mask = words.ne(self.WORD.pad_index)
                words_len = torch.sum(mask, dim=-1)

                # loss = self.model.dynamic_loss4(words, words_len, feats, edges,
                #                                 labels, p)

                # 6,7是分析实验
                loss = self.model.dynamic_loss4(words, words_len, feats, edges,
                                                labels, p)
                # num_all += n_all
                # num_false += n_false
                # all_pred_step += num_follow_pred_step
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.args.clip)
                self.optimizer.step()
                self.scheduler.step()

                # edge_preds, label_preds = self.model.decode(s_edge, s_label)
                # metric(label_preds.masked_fill(~(edge_preds.gt(0) & mask), -1),
                #        labels.masked_fill(~(edges.gt(0) & mask), -1))
                bar.set_postfix_str(
                    f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}"
                )
            # return num_false / num_all

    def _train(self, loader, mode):
        self.model.train()
        bar, metric = progress_bar(loader), ChartMetric()
        if (mode in ('dual', 'par-dual')):
            for words, *feats, edges, labels, transitions, translabel, states in bar:
                self.optimizer.zero_grad()
                # mask = words.ne(self.WORD.pad_index)
                # mask = mask.unsqueeze(1) & mask.unsqueeze(2)
                # mask[:, 0] = 0
                transition_mask = transitions.ge(0)
                transition_len = transition_mask.sum(1)
                # pdb.set_trace()
                action_score, label_score = self.model(words, feats, states,
                                                       transitions,
                                                       transition_len)
                loss = self.model.dual_loss(action_score, label_score,
                                            transitions, translabel)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.args.clip)
                self.optimizer.step()
                self.scheduler.step()

                # edge_preds, label_preds = self.model.decode(s_edge, s_label)
                # metric(label_preds.masked_fill(~(edge_preds.gt(0) & mask), -1),
                #        labels.masked_fill(~(edges.gt(0) & mask), -1))
                bar.set_postfix_str(
                    f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}"
                )
        else:
            for words, *feats, edges, labels, transitions, states in bar:
                self.optimizer.zero_grad()

                # mask = words.ne(self.WORD.pad_index)
                # mask = mask.unsqueeze(1) & mask.unsqueeze(2)
                # mask[:, 0] = 0
                transition_mask = transitions.ge(0)
                transition_len = transition_mask.sum(1)
                score = self.model(words, feats, states, transitions,
                                   transition_len)
                loss = self.model.loss(score, transitions)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.args.clip)
                self.optimizer.step()
                self.scheduler.step()

                # edge_preds, label_preds = self.model.decode(s_edge, s_label)
                # metric(label_preds.masked_fill(~(edge_preds.gt(0) & mask), -1),
                #        labels.masked_fill(~(edges.gt(0) & mask), -1))
                bar.set_postfix_str(
                    f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}"
                )

    @torch.no_grad()
    def _batch_evaluate(self, loader, mode):
        self.model.eval()

        total_loss, metric = 0, ChartMetric()

        if (mode == 'par-dual'):
            for words, *feats, edges, labels, transitions, translabel, states in loader:
                mask = words.ne(self.WORD.pad_index)
                words_len = torch.sum(mask, dim=-1)
                mask = mask.unsqueeze(1) & mask.unsqueeze(2)
                mask[:, 0] = 0
                edge_preds, label_preds = self.model.batch_decode_2(
                    words, words_len, feats)
                metric(label_preds.masked_fill(~(edge_preds.gt(0) & mask), -1),
                       labels.masked_fill(~(edges.gt(0) & mask), -1))
            # total_loss /= len(loader)

            return metric

        elif (mode == 'dual'):
            for words, *feats, edges, labels, transitions, translabel, states in loader:
                mask = words.ne(self.WORD.pad_index)
                words_len = torch.sum(mask, dim=-1)
                mask = mask.unsqueeze(1) & mask.unsqueeze(2)
                mask[:, 0] = 0
                edge_preds, label_preds = self.model.batch_decode(
                    words, words_len, feats)
                metric(label_preds.masked_fill(~(edge_preds.gt(0) & mask), -1),
                       labels.masked_fill(~(edges.gt(0) & mask), -1))
            # total_loss /= len(loader)
            return metric

        else:
            # for single mlp
            for words, *feats, edges, labels, transitions, states in loader:
                mask = words.ne(self.WORD.pad_index)
                words_len = torch.sum(mask, dim=-1)
                mask = mask.unsqueeze(1) & mask.unsqueeze(2)
                mask[:, 0] = 0
                edge_preds, label_preds = self.model.batch_decode_3(
                    words, words_len, feats)
                metric(label_preds.masked_fill(~(edge_preds.gt(0) & mask), -1),
                       labels.masked_fill(~(edges.gt(0) & mask), -1))
            # total_loss /= len(loader)
            return metric

    @torch.no_grad()
    def _evaluate(self, loader, mode):
        # for not use batch decode
        self.model.eval()

        total_loss, metric = 0, ChartMetric()

        if (mode not in ('dual', 'par-dual')):
            for words, *feats, edges, labels, transitions, states in loader:
                mask = words.ne(self.WORD.pad_index)
                words_len = torch.sum(mask, dim=-1)
                mask = mask.unsqueeze(1) & mask.unsqueeze(2)
                mask[:, 0] = 0
                # s_edge, s_label = self.model(words, feats)
                # loss = self.model.loss(s_edge, s_label, edges, labels, mask)
                # total_loss += loss.item()

                edge_preds, label_preds = self.model.decode(
                    words, words_len, feats)
                metric(label_preds.masked_fill(~(edge_preds.gt(0) & mask), -1),
                       labels.masked_fill(~(edges.gt(0) & mask), -1))
            # total_loss /= len(loader)
            return metric

        else:
            for words, *feats, edges, labels, transitions, translabel, states in loader:
                mask = words.ne(self.WORD.pad_index)
                words_len = torch.sum(mask, dim=-1)
                mask = mask.unsqueeze(1) & mask.unsqueeze(2)
                mask[:, 0] = 0
                # s_edge, s_label = self.model(words, feats)
                # loss = self.model.loss(s_edge, s_label, edges, labels, mask)
                # total_loss += loss.item()
                edge_preds, label_preds = self.model.decode(
                    words, words_len, feats)
                # edge_preds, label_preds = self.model.decode(
                #     words, words_len, feats)
                # print('no batch')

                metric(label_preds.masked_fill(~(edge_preds.gt(0) & mask), -1),
                       labels.masked_fill(~(edges.gt(0) & mask), -1))
            # total_loss /= len(loader)

            return metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds = {}
        charts, probs = [], []
        for words, *feats in progress_bar(loader):
            mask = words.ne(self.WORD.pad_index)
            words_len = torch.sum(mask, dim=-1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            lens = mask[:, 1].sum(-1).tolist()
            # s_edge, s_label = self.model(words, feats)
            # edge_preds, label_preds = self.model.decode(s_edge, s_label)
            if(self.args.decode_mode == 'mlp'):
                edge_preds, label_preds = self.model.batch_decode_3(
                    words, words_len, feats)
            elif(self.args.decode_mode == 'dual'):
                edge_preds, label_preds = self.model.batch_decode(
                    words, words_len, feats)
            chart_preds = label_preds.masked_fill(~(edge_preds.gt(0) & mask),
                                                  -1)
            charts.extend(chart[1:i, :i].tolist()
                          for i, chart in zip(lens, chart_preds.unbind()))
            # if self.args.prob:
            #     probs.extend([
            #         prob[1:i, :i].cpu()
            #         for i, prob in zip(lens,
            #                            s_edge.softmax(-1).unbind())
            #     ])
        charts = [
            CoNLL.build_relations(
                [[self.LABEL.vocab[i] if i >= 0 else None for i in row]
                 for row in chart]) for chart in charts
        ]
        preds = {'labels': charts}
        if self.args.prob:
            preds['probs'] = probs

        return preds

    @classmethod
    def build(cls,
              path,
              optimizer_args={
                  'lr': 1e-3,
                  'betas': (.0, .95),
                  'eps': 1e-12,
                  'weight_decay': 3e-9
              },
              scheduler_args={'gamma': .75**(1 / 5000)},
              min_freq=7,
              fix_len=20,
              **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            optimizer_args (dict):
                Arguments for creating an optimizer.
            scheduler_args (dict):
                Arguments for creating a scheduler.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default:7.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # pdb.set_trace()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Building the fields")
        WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
        TAG, CHAR, LEMMA, BERT = None, None, None, None
        if 'tag' in args.feat:
            TAG = Field('tags', bos=bos)
        if 'char' in args.feat:
            CHAR = SubwordField('chars',
                                pad=pad,
                                unk=unk,
                                bos=bos,
                                fix_len=args.fix_len)
        if 'lemma' in args.feat:
            LEMMA = Field('lemmas', pad=pad, unk=unk, bos=bos, lower=True)
        if 'bert' in args.feat:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.bert)
            BERT = SubwordField('bert',
                                pad=tokenizer.pad_token,
                                unk=tokenizer.unk_token,
                                bos=tokenizer.bos_token or tokenizer.cls_token,
                                fix_len=args.fix_len,
                                tokenize=tokenizer.tokenize)
            BERT.vocab = tokenizer.get_vocab()
        EDGE = ChartField('edges', use_vocab=False, fn=CoNLL.get_edges)
        LABEL = ChartField('labels', fn=CoNLL.get_labels)
        if (args.decode_mode not in ('dual', 'par-dual')):
            Transition = TransitionField('transitions', label_field=LABEL)
            State = StateField('states',
                               label_field=LABEL,
                               transition_field=Transition)

            transform = CoNLL(FORM=(WORD, CHAR, BERT),
                              LEMMA=LEMMA,
                              POS=TAG,
                              PHEAD=(EDGE, LABEL, Transition, State))
        else:
            Transition = ParTransitionField('transitions', label_field=LABEL)
            State = ParStateField('states',
                                  label_field=LABEL,
                                  transition_field=Transition)
            TransLabel = TranLabelField('translabel', label_field=LABEL)
            transform = CoNLL(FORM=(WORD, CHAR, BERT),
                              LEMMA=LEMMA,
                              POS=TAG,
                              PHEAD=(EDGE, LABEL, Transition, TransLabel,
                                     State))

        train = Dataset(transform, args.train)
        WORD.build(
            train, args.min_freq,
            (Embedding.load(args.embed, args.unk) if args.embed else None))
        if TAG is not None:
            TAG.build(train)
        if CHAR is not None:
            CHAR.build(train)
        if LEMMA is not None:
            LEMMA.build(train)
        LABEL.build(train)
        Transition.build(train)

        args.update({
            'n_words': WORD.vocab.n_init,
            'n_labels': len(LABEL.vocab),
            'n_tags': len(TAG.vocab) if TAG is not None else None,
            'n_chars': len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index': CHAR.pad_index if CHAR is not None else None,
            'n_lemmas': len(LEMMA.vocab) if LEMMA is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'window': args.window,
            'loss_type': args.loss_type,
            'n_transitions': len(Transition.vocab),
            'transition_vocab': Transition.vocab,
            'n_transition_embed': 600
        })
        # pdb.set_trace()
        logger.info(f"{transform}")

        logger.info("Building the model")

        model = cls.MODEL(**args).load_pretrained(WORD.embed).to(args.device)
        logger.info(f"{model}\n")

        optimizer = Adam(model.parameters(), **optimizer_args)
        scheduler = ExponentialLR(optimizer, **scheduler_args)

        return cls(args, model, transform, optimizer, scheduler)
