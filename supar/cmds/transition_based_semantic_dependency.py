# -*- coding: utf-8 -*-

import argparse

from supar import TransitionSemanticDependencyParser
from supar.cmds.cmd import parse


def main():
    parser = argparse.ArgumentParser(
        description='Create Transition-based Semantic Dependency Parser.')
    parser.set_defaults(Parser=TransitionSemanticDependencyParser)
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument(
        '--feat',
        '-f',
        default='tag,char,lemma',
        help='additional features to use，separated by commas.')

    subparser.add_argument('--doloss',
                           default='formal',
                           type=str,
                           help='the dynamic loss mode')

    subparser.add_argument('--pro',
                           default=0.6,
                           type=float,
                           help='the probablity')
    
    subparser.add_argument('--itp',
                           default=0.5,
                           type=float,
                           help='the interploation')

    subparser.add_argument('--k',
                           default=3,
                           type=int,
                           help='the num of epoch which donot use dynamic')

    subparser.add_argument('--mu',
                           default=0,
                           type=int,
                           help='hyper-paramter of pro to change')

    subparser.add_argument('--batch_train',
                           action='store_true',
                           help='whether to use dynamic batch train')

    subparser.add_argument('--dynamic',
                           action='store_true',
                           help='whether to use the dynamic oracle')
    subparser.add_argument('--build',
                           '-b',
                           action='store_true',
                           help='whether to build the model first')
    subparser.add_argument('--max-len',
                           type=int,
                           help='max length of the sentences')
    subparser.add_argument('--buckets',
                           default=32,
                           type=int,
                           help='max num of buckets to use')
    subparser.add_argument('--train',
                           default='data/sdp/DM/train.conllu',
                           help='path to train file')
    subparser.add_argument('--dev',
                           default='data/sdp/DM/dev.conllu',
                           help='path to dev file')
    subparser.add_argument('--test',
                           default='data/sdp/DM/test.conllu',
                           help='path to test file')
    subparser.add_argument('--embed',
                           default='data/glove.6B.100d.txt',
                           help='path to pretrained embeddings')
    subparser.add_argument('--unk',
                           default='unk',
                           help='unk token in pretrained embeddings')
    subparser.add_argument('--n-embed',
                           default=100,
                           type=int,
                           help='dimension of embeddings')
    subparser.add_argument('--bert',
                           default='bert-base-cased',
                           help='which bert model to use')

    subparser.add_argument('--window',
                           default=1,
                           type=int,
                           help='num of cell feature')

    # subparser.add_argument('--decode_mode',
    #                        default='dual',
    #                        help='the decoder to use: mlp, att, lstm, beta, dual')

    subparser.add_argument(
        '--loss_type',
        default='Formal',
        help='the used loss type: CrossEntropyLoss, MLL(multilabelloss)')

    # evaluate
    subparser = subparsers.add_parser(
        'evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--buckets',
                           default=8,
                           type=int,
                           help='max num of buckets to use')
    subparser.add_argument('--data',
                           default='data/sdp/DM/test.conllu',
                           help='path to dataset')

    subparser.add_argument('--batch_decode',
                           action='store_true',
                           help='whether to use batch_decode')

    subparser.add_argument(
        '--feat',
        '-f',
        default='tag,char,lemma',
        help='additional features to use，separated by commas.')

    # predict
    subparser = subparsers.add_parser(
        'predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--prob',
                           action='store_true',
                           help='whether to output probs')
    subparser.add_argument('--buckets',
                           default=8,
                           type=int,
                           help='max num of buckets to use')
    subparser.add_argument('--data',
                           default='data/sdp/DM/test.conllu',
                           help='path to dataset')
    subparser.add_argument('--pred',
                           default='pred.conllu',
                           help='path to predicted result')
    parse(parser)


if __name__ == "__main__":
    main()
