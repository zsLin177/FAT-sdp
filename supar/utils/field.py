# -*- coding: utf-8 -*-

from collections import Counter

import torch
from supar.utils.fn import pad
from supar.utils.vocab import Vocab
import pdb


class RawField(object):
    r"""
    Defines a general datatype.

    A :class:`RawField` object does not assume any property of the datatype and
    it holds parameters relating to how a datatype should be processed.

    Args:
        name (str):
            The name of the field.
        fn (function):
            The function used for preprocessing the examples. Default: ``None``.
    """
    def __init__(self, name, fn=None):
        self.name = name
        self.fn = fn

    def __repr__(self):
        return f"({self.name}): {self.__class__.__name__}()"

    def preprocess(self, sequence):
        return self.fn(sequence) if self.fn is not None else sequence

    def transform(self, sequences):
        return [self.preprocess(seq) for seq in sequences]

    def compose(self, sequences):
        return sequences


class Field(RawField):
    r"""
    Defines a datatype together with instructions for converting to :class:`~torch.Tensor`.
    :class:`Field` models common text processing datatypes that can be represented by tensors.
    It holds a :class:`Vocab` object that defines the set of possible values
    for elements of the field and their corresponding numerical representations.
    The :class:`Field` object also holds other parameters relating to how a datatype
    should be numericalized, such as a tokenization method.

    Args:
        name (str):
            The name of the field.
        pad_token (str):
            The string token used as padding. Default: ``None``.
        unk_token (str):
            The string token used to represent OOV words. Default: ``None``.
        bos_token (str):
            A token that will be prepended to every example using this field, or ``None`` for no `bos_token`.
            Default: ``None``.
        eos_token (str):
            A token that will be appended to every example using this field, or ``None`` for no `eos_token`.
        lower (bool):
            Whether to lowercase the text in this field. Default: ``False``.
        use_vocab (bool):
            Whether to use a :class:`Vocab` object. If ``False``, the data in this field should already be numerical.
            Default: ``True``.
        tokenize (function):
            The function used to tokenize strings using this field into sequential examples. Default: ``None``.
        fn (function):
            The function used for preprocessing the examples. Default: ``None``.
    """
    def __init__(self,
                 name,
                 pad=None,
                 unk=None,
                 bos=None,
                 eos=None,
                 lower=False,
                 use_vocab=True,
                 tokenize=None,
                 fn=None):
        self.name = name
        self.pad = pad
        self.unk = unk
        self.bos = bos
        self.eos = eos
        self.lower = lower
        self.use_vocab = use_vocab
        self.tokenize = tokenize
        self.fn = fn

        self.specials = [
            token for token in [pad, unk, bos, eos] if token is not None
        ]

    def __repr__(self):
        s, params = f"({self.name}): {self.__class__.__name__}(", []
        if self.pad is not None:
            params.append(f"pad={self.pad}")
        if self.unk is not None:
            params.append(f"unk={self.unk}")
        if self.bos is not None:
            params.append(f"bos={self.bos}")
        if self.eos is not None:
            params.append(f"eos={self.eos}")
        if self.lower:
            params.append(f"lower={self.lower}")
        if not self.use_vocab:
            params.append(f"use_vocab={self.use_vocab}")
        s += ", ".join(params)
        s += ")"

        return s

    def __getstate__(self):
        state = self.__dict__
        if self.tokenize is None:
            state['tokenize_args'] = None
        elif self.tokenize.__module__.startswith('transformers'):
            state['tokenize_args'] = (self.tokenize.__module__,
                                      self.tokenize.__self__.name_or_path)
            state['tokenize'] = None
        return state

    def __setstate__(self, state):
        tokenize_args = state.pop('tokenize_args', None)
        if tokenize_args is not None and tokenize_args[0].startswith(
                'transformers'):
            from transformers import AutoTokenizer
            state['tokenize'] = AutoTokenizer.from_pretrained(
                tokenize_args[1]).tokenize
        self.__dict__.update(state)

    @property
    def pad_index(self):
        if self.pad is None:
            return 0
        if hasattr(self, 'vocab'):
            return self.vocab[self.pad]
        return self.specials.index(self.pad)

    @property
    def unk_index(self):
        if self.unk is None:
            return 0
        if hasattr(self, 'vocab'):
            return self.vocab[self.unk]
        return self.specials.index(self.unk)

    @property
    def bos_index(self):
        if hasattr(self, 'vocab'):
            return self.vocab[self.bos]
        return self.specials.index(self.bos)

    @property
    def eos_index(self):
        if hasattr(self, 'vocab'):
            return self.vocab[self.eos]
        return self.specials.index(self.eos)

    @property
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def preprocess(self, sequence):
        r"""
        Loads a single example using this field, tokenizing if necessary.
        The sequence will be first passed to ``fn`` if available.
        If ``tokenize`` is not None, the input will be tokenized.
        Then the input will be lowercased optionally.

        Args:
            sequence (list):
                The sequence to be preprocessed.

        Returns:
            A list of preprocessed sequence.
        """

        if self.fn is not None:
            sequence = self.fn(sequence)
        if self.tokenize is not None:
            sequence = self.tokenize(sequence)
        if self.lower:
            sequence = [str.lower(token) for token in sequence]

        return sequence

    def build(self, dataset, min_freq=1, embed=None):
        r"""
        Constructs a :class:`Vocab` object for this field from the dataset.
        If the vocabulary has already existed, this function will have no effect.

        Args:
            dataset (Dataset):
                A :class:`Dataset` object. One of the attributes should be named after the name of this field.
            min_freq (int):
                The minimum frequency needed to include a token in the vocabulary. Default: 1.
            embed (Embedding):
                An Embedding object, words in which will be extended to the vocabulary. Default: ``None``.
        """

        if hasattr(self, 'vocab'):
            return
        sequences = getattr(dataset, self.name)
        counter = Counter(token for seq in sequences
                          for token in self.preprocess(seq))
        self.vocab = Vocab(counter, min_freq, self.specials, self.unk_index)

        if not embed:
            self.embed = None
        else:
            tokens = self.preprocess(embed.tokens)
            # if the `unk` token has existed in the pretrained,
            # then replace it with a self-defined one
            if embed.unk:
                tokens[embed.unk_index] = self.unk

            self.vocab.extend(tokens)
            self.embed = torch.zeros(len(self.vocab), embed.dim)
            self.embed[self.vocab[tokens]] = embed.vectors
            self.embed /= torch.std(self.embed)

    def transform(self, sequences):
        r"""
        Turns a list of sequences that use this field into tensors.

        Each sequence is first preprocessed and then numericalized if needed.

        Args:
            sequences (list[list[str]]):
                A list of sequences.

        Returns:
            A list of tensors transformed from the input sequences.
        """

        sequences = [self.preprocess(seq) for seq in sequences]
        if self.use_vocab:
            sequences = [self.vocab[seq] for seq in sequences]
        if self.bos:
            sequences = [[self.bos_index] + seq for seq in sequences]
        if self.eos:
            sequences = [seq + [self.eos_index] for seq in sequences]
        sequences = [torch.tensor(seq) for seq in sequences]

        return sequences

    def compose(self, sequences):
        r"""
        Composes a batch of sequences into a padded tensor.

        Args:
            sequences (list[~torch.Tensor]):
                A list of tensors.

        Returns:
            A padded tensor converted to proper device.
        """

        return pad(sequences, self.pad_index).to(self.device)


class SubwordField(Field):
    r"""
    A field that conducts tokenization and numericalization over each token rather the sequence.

    This is customized for models requiring character/subword-level inputs, e.g., CharLSTM and BERT.

    Args:
        fix_len (int):
            A fixed length that all subword pieces will be padded to.
            This is used for truncating the subword pieces that exceed the length.
            To save the memory, the final length will be the smaller value
            between the max length of subword pieces in a batch and `fix_len`.

    Examples:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        >>> field = SubwordField('bert',
                                 pad=tokenizer.pad_token,
                                 unk=tokenizer.unk_token,
                                 bos=tokenizer.cls_token,
                                 eos=tokenizer.sep_token,
                                 fix_len=20,
                                 tokenize=tokenizer.tokenize)
        >>> field.vocab = tokenizer.get_vocab()  # no need to re-build the vocab
        >>> field.transform([['This', 'field', 'performs', 'token-level', 'tokenization']])[0]
        tensor([[  101,     0,     0],
                [ 1188,     0,     0],
                [ 1768,     0,     0],
                [10383,     0,     0],
                [22559,   118,  1634],
                [22559,  2734,     0],
                [  102,     0,     0]])
    """
    def __init__(self, *args, **kwargs):
        self.fix_len = kwargs.pop('fix_len') if 'fix_len' in kwargs else 0
        super().__init__(*args, **kwargs)

    def build(self, dataset, min_freq=1, embed=None):
        if hasattr(self, 'vocab'):
            return
        sequences = getattr(dataset, self.name)
        counter = Counter(piece for seq in sequences for token in seq
                          for piece in self.preprocess(token))
        self.vocab = Vocab(counter, min_freq, self.specials, self.unk_index)

        if not embed:
            self.embed = None
        else:
            tokens = self.preprocess(embed.tokens)
            # if the `unk` token has existed in the pretrained,
            # then replace it with a self-defined one
            if embed.unk:
                tokens[embed.unk_index] = self.unk

            self.vocab.extend(tokens)
            self.embed = torch.zeros(len(self.vocab), embed.dim)
            self.embed[self.vocab[tokens]] = embed.vectors

    def transform(self, sequences):
        sequences = [[self.preprocess(token) for token in seq]
                     for seq in sequences]
        if self.fix_len <= 0:
            self.fix_len = max(
                len(token) for seq in sequences for token in seq)
        if self.use_vocab:
            sequences = [[[self.vocab[i]
                           for i in token] if token else [self.unk_index]
                          for token in seq] for seq in sequences]
        if self.bos:
            sequences = [[[self.bos_index]] + seq for seq in sequences]
        if self.eos:
            sequences = [seq + [[self.eos_index]] for seq in sequences]
        lens = [
            min(self.fix_len, max(len(ids) for ids in seq))
            for seq in sequences
        ]
        sequences = [
            pad([torch.tensor(ids[:i]) for ids in seq], self.pad_index, i)
            for i, seq in zip(lens, sequences)
        ]

        return sequences


class ChartField(Field):
    r"""
    Field dealing with chart inputs.

    Examples:
        >>> chart = [[    None,    'NP',    None,    None,  'S|<>',     'S'],
                     [    None,    None, 'VP|<>',    None,    'VP',    None],
                     [    None,    None,    None, 'VP|<>',  'S+VP',    None],
                     [    None,    None,    None,    None,    'NP',    None],
                     [    None,    None,    None,    None,    None,  'S|<>'],
                     [    None,    None,    None,    None,    None,    None]]
        >>> field.transform([chart])[0]
        tensor([[ -1,  37,  -1,  -1, 107,  79],
                [ -1,  -1, 120,  -1, 112,  -1],
                [ -1,  -1,  -1, 120,  86,  -1],
                [ -1,  -1,  -1,  -1,  37,  -1],
                [ -1,  -1,  -1,  -1,  -1, 107],
                [ -1,  -1,  -1,  -1,  -1,  -1]])
    """
    def build(self, dataset, min_freq=1):
        counter = Counter(i for chart in getattr(dataset, self.name)
                          for row in self.preprocess(chart) for i in row
                          if i is not None)

        self.vocab = Vocab(counter, min_freq, self.specials, self.unk_index)

    def transform(self, charts):
        charts = [self.preprocess(chart) for chart in charts]
        if self.use_vocab:
            charts = [[[self.vocab[i] if i is not None else -1 for i in row]
                       for row in chart] for chart in charts]
        charts = [torch.tensor(chart) for chart in charts]
        return charts


def get_oracle_states(annotated_sentence_len, directed_arc_indices, arc_tag):
    """
    根据标注了SDG的句子，生成正确的转移序列。
    :param annotated_sentence_len:tokens列表的长度（包括root）
    :param directed_arc_indices:有向依存弧列表
    :param arc_tag:依存弧标签列表
    :return:state序列
    """
    graph = {}
    for token_idx in range(annotated_sentence_len):
        graph[token_idx] = []

    # 构建字典形式存储的语义依存图
    # 字典的键值对含义为：(孩子节点:[(头节点_1，弧标签_1),(头节点_2，弧标签_2)...])
    for arc, arc_tag in zip(directed_arc_indices, arc_tag):
        graph[arc[0]].append((arc[1], arc_tag))

    # N为节点个数，其中包含一个根节点ROOT
    N = len(graph)

    # 以列表形式存储的自顶向下的语义依存图，top_down_graph[i]为索引为i的节点所有孩子节点组成的列表
    top_down_graph = [[] for i in range(N)
                      ]  # N-1 real point, 1 root point => N point

    # sub_graph[i][j]表示索引为j的节点作为头节点，索引为i的节点作为孩子节点时，二者之间是否存在子图结构（连通）
    sub_graph = [[False for i in range(N)] for j in range(N)]

    # 生成top_down_graph
    for i in range(N):
        for head_tuple_of_point_i in graph[i]:
            head = head_tuple_of_point_i[0]
            top_down_graph[head].append(i)

    result_states = []
    actions = []  # 动作序列
    stack = [0]  # 正在处理的节点序列
    buffer = []  # 待处理的节点序列
    deque = []  # 暂时跳过的节点序列（可能存在多个头节点）

    # 待处理节点进入buffer
    for i in range(N - 1, 0, -1):
        buffer.append(i)

    def has_head(w0, w1):
        """
        :param w0: 节点索引
        :param w1: 节点索引
        :return: w1是否为w0的头节点
        """
        if w0 <= 0:
            return False
        for arc_tuple in graph[w0]:
            if arc_tuple[0] == w1:
                return True
        return False

    def has_unfound_child(w):
        """
        :param w: 节点索引
        :return: w是否还有未找到的孩子节点
        """
        for child in top_down_graph[w]:
            if not sub_graph[child][w]:
                return True
        return False

    def has_other_head(w):
        """
        :param w: 节点索引
        :return: w除了当前节点外是否还有其余头节点
        """
        if w <= 0:
            return False
        head_num = 0
        for h in sub_graph[w]:
            if h:
                head_num += 1
        if head_num + 1 < len(graph[w]):
            return True
        return False

    def lack_head(w):
        """
        :param w: 节点索引
        :return: w是否还有未找到的头节点
        """
        if w <= 0:
            return False
        head_num = 0
        for h in sub_graph[w]:
            if h:
                head_num += 1
        if head_num < len(graph[w]):
            return True
        return False

    def has_other_child_in_stack(w):
        """
        :param w: 节点索引
        :return: 除了栈顶节点外，w是否在栈中还有其余孩子节点
        """
        if w <= 0:
            return False
        for c in top_down_graph[w]:
            if c in stack and c != stack[-1] and not sub_graph[c][w]:
                return True
        return False

    def has_other_head_in_stack(w):
        """
        :param w: 节点索引
        :return: 除了栈顶节点外，w是否在栈中还有其余头节点
        """
        if w <= 0:
            return False
        for h in graph[w]:
            if h[0] in stack and h[0] != stack[-1] and not sub_graph[w][h[0]]:
                return True
        return False

    def get_arc_label(w0, w1):
        """
        :param w0: 节点索引
        :param w1: 节点索引
        :return: w1作为头节点，w0作为孩子节点时，依存弧的标签
        """
        for h in graph[w0]:
            if h[0] == w1:
                return h[1]

    def get_oracle_states_onestep():
        """
        根据当前stack、buffer、deque、actions四个部分，生成下一步的转移动作
        """

        result_states.append([stack[:: -1], buffer[:: -1], deque[:: -1], actions[:: -1]])
        # 栈顶从下标0开始,buffer的头从下标0开始,最近加入deque的元素从下标0开始,最近的action从下标0开始
        # 我把最要关注的都放在了前面

        b0 = buffer[-1] if len(buffer) > 0 else -1
        s0 = stack[-1] if len(stack) > 0 else -1

        # buffer首节点与栈顶节点有关系

        # 栈顶节点是buffer首节点的孩子节点，即生成弧的动作是"Left"
        if s0 > 0 and has_head(s0, b0):
            # 栈顶节点没有未找到的孩子节点或其余头节点，则直接将其出栈，执行"Left-Reduce"操作
            if not has_unfound_child(s0) and not has_other_head(s0):
                actions.append("LR:" + get_arc_label(s0, b0))
                stack.pop()
                sub_graph[s0][b0] = True
                return
            # 否则需要将栈顶节点暂时入deque保存，以便之后重新进栈，执行"Left-Pass"操作。
            else:
                actions.append("LP:" + get_arc_label(s0, b0))
                deque.append(stack.pop())
                sub_graph[s0][b0] = True
                return

        # buffer首节点是栈顶节点的孩子节点，即生成弧的动作是"Right"
        elif s0 >= 0 and has_head(b0, s0):
            # buffer首节点在栈中除了栈顶节点以外，没有其他的孩子节点或者头节点，则将其进栈处理，执行"Right-Shift"操作
            if not has_other_child_in_stack(
                    b0) and not has_other_head_in_stack(b0):
                actions.append("RS:" + get_arc_label(b0, s0))
                # Shift操作前，要将deque中暂存的节点先压栈
                while len(deque) != 0:
                    stack.append(deque.pop())
                stack.append(buffer.pop())
                sub_graph[b0][s0] = True
                return

            # buffer首节点在栈中除了栈顶节点以外，还有其他的孩子节点或者头节点，则将其暂时入deque保存，执行"Right-Pass"操作
            elif s0 > 0:
                actions.append("RP:" + get_arc_label(b0, s0))
                deque.append(stack.pop())
                sub_graph[b0][s0] = True
                return

        # buffer首节点与栈顶节点无关系，生成弧动作为"None"

        # buffer首节点在栈中除了栈顶节点以外，没有其他的孩子节点或者头节点，则将其进栈处理，执行"None-Shift"操作
        elif len(buffer) != 0 and not has_other_head_in_stack(
                b0) and not has_other_child_in_stack(b0):
            actions.append("NS")
            # Shift操作前，要将deque中暂存的节点先压栈
            while len(deque) != 0:
                stack.append(deque.pop())
            stack.append(buffer.pop())
            return

        # 栈顶节点没有未找到的孩子节点或头节点，说明完成了所有依存关系的生成，可以出栈丢弃了，执行"None-Reduce"操作
        elif s0 > 0 and not has_unfound_child(s0) and not lack_head(s0):
            actions.append("NR")
            stack.pop()
            return

        # 栈顶节点还有未找到的孩子节点或头节点，则将其暂时入deque保存，执行"None-Pass"操作
        elif s0 > 0:
            actions.append("NP")
            deque.append(stack.pop())
            return

        # 如果出现了意料之外的分支，那么就说明出错了
        else:
            actions.append('-E-')
            print('"error in oracle!"')
            return

    # 每次生成一步转移动作，终止条件为：buffer为空
    while len(buffer) != 0:
        get_oracle_states_onestep()

    max_len = max(len(actions) - 1, annotated_sentence_len)
    for state in result_states:
        for i in range(len(state)):
            state[i] = state[i] + [-1] * (max_len - len(state[i]))

    return result_states


def get_oracle_actions(annotated_sentence_len, directed_arc_indices, arc_tag):
    """
    根据标注了SDG的句子，生成正确的转移序列。
    :param annotated_sentence_len:tokens列表的长度（包括root）
    :param directed_arc_indices:有向依存弧列表
    :param arc_tag:依存弧标签列表
    :return:转移动作序列
    """
    graph = {}
    for token_idx in range(annotated_sentence_len):
        graph[token_idx] = []

    # 构建字典形式存储的语义依存图
    # 字典的键值对含义为：(孩子节点:[(头节点_1，弧标签_1),(头节点_2，弧标签_2)...])
    for arc, arc_tag in zip(directed_arc_indices, arc_tag):
        graph[arc[0]].append((arc[1], arc_tag))

    # N为节点个数，其中包含一个根节点ROOT
    N = len(graph)

    # 以列表形式存储的自顶向下的语义依存图，top_down_graph[i]为索引为i的节点所有孩子节点组成的列表
    top_down_graph = [[] for i in range(N)
                      ]  # N-1 real point, 1 root point => N point

    # sub_graph[i][j]表示索引为j的节点作为头节点，索引为i的节点作为孩子节点时，二者之间是否存在子图结构（连通）
    sub_graph = [[False for i in range(N)] for j in range(N)]

    # 生成top_down_graph
    for i in range(N):
        for head_tuple_of_point_i in graph[i]:
            head = head_tuple_of_point_i[0]
            top_down_graph[head].append(i)

    actions = []  # 动作序列
    stack = [0]  # 正在处理的节点序列
    buffer = []  # 待处理的节点序列
    deque = []  # 暂时跳过的节点序列（可能存在多个头节点）

    # 待处理节点进入buffer
    for i in range(N - 1, 0, -1):
        buffer.append(i)

    def has_head(w0, w1):
        """
        :param w0: 节点索引
        :param w1: 节点索引
        :return: w1是否为w0的头节点
        """
        if w0 <= 0:
            return False
        for arc_tuple in graph[w0]:
            if arc_tuple[0] == w1:
                return True
        return False

    def has_unfound_child(w):
        """
        :param w: 节点索引
        :return: w是否还有未找到的孩子节点
        """
        for child in top_down_graph[w]:
            if not sub_graph[child][w]:
                return True
        return False

    def has_other_head(w):
        """
        :param w: 节点索引
        :return: w除了当前节点外是否还有其余头节点
        """
        if w <= 0:
            return False
        head_num = 0
        for h in sub_graph[w]:
            if h:
                head_num += 1
        if head_num + 1 < len(graph[w]):
            return True
        return False

    def lack_head(w):
        """
        :param w: 节点索引
        :return: w是否还有未找到的头节点
        """
        if w <= 0:
            return False
        head_num = 0
        for h in sub_graph[w]:
            if h:
                head_num += 1
        if head_num < len(graph[w]):
            return True
        return False

    def has_other_child_in_stack(w):
        """
        :param w: 节点索引
        :return: 除了栈顶节点外，w是否在栈中还有其余孩子节点
        """
        if w <= 0:
            return False
        for c in top_down_graph[w]:
            if c in stack and c != stack[-1] and not sub_graph[c][w]:
                return True
        return False

    def has_other_head_in_stack(w):
        """
        :param w: 节点索引
        :return: 除了栈顶节点外，w是否在栈中还有其余头节点
        """
        if w <= 0:
            return False
        for h in graph[w]:
            if h[0] in stack and h[0] != stack[-1] and not sub_graph[w][h[0]]:
                return True
        return False

    def get_arc_label(w0, w1):
        """
        :param w0: 节点索引
        :param w1: 节点索引
        :return: w1作为头节点，w0作为孩子节点时，依存弧的标签
        """
        for h in graph[w0]:
            if h[0] == w1:
                return h[1]

    def get_oracle_actions_onestep():
        """
        根据当前stack、buffer、deque、actions四个部分，生成下一步的转移动作
        """



        b0 = buffer[-1] if len(buffer) > 0 else -1
        s0 = stack[-1] if len(stack) > 0 else -1

        # buffer首节点与栈顶节点有关系

        # 栈顶节点是buffer首节点的孩子节点，即生成弧的动作是"Left"
        if s0 > 0 and has_head(s0, b0):
            # 栈顶节点没有未找到的孩子节点或其余头节点，则直接将其出栈，执行"Left-Reduce"操作
            if not has_unfound_child(s0) and not has_other_head(s0):
                actions.append("LR:" + get_arc_label(s0, b0))
                stack.pop()
                sub_graph[s0][b0] = True
                return
            # 否则需要将栈顶节点暂时入deque保存，以便之后重新进栈，执行"Left-Pass"操作。
            else:
                actions.append("LP:" + get_arc_label(s0, b0))
                deque.append(stack.pop())
                sub_graph[s0][b0] = True
                return

        # buffer首节点是栈顶节点的孩子节点，即生成弧的动作是"Right"
        elif s0 >= 0 and has_head(b0, s0):
            # buffer首节点在栈中除了栈顶节点以外，没有其他的孩子节点或者头节点，则将其进栈处理，执行"Right-Shift"操作
            if not has_other_child_in_stack(
                    b0) and not has_other_head_in_stack(b0):
                actions.append("RS:" + get_arc_label(b0, s0))
                # Shift操作前，要将deque中暂存的节点先压栈
                while len(deque) != 0:
                    stack.append(deque.pop())
                stack.append(buffer.pop())
                sub_graph[b0][s0] = True
                return

            # buffer首节点在栈中除了栈顶节点以外，还有其他的孩子节点或者头节点，则将其暂时入deque保存，执行"Right-Pass"操作
            elif s0 > 0:
                actions.append("RP:" + get_arc_label(b0, s0))
                deque.append(stack.pop())
                sub_graph[b0][s0] = True
                return

        # buffer首节点与栈顶节点无关系，生成弧动作为"None"

        # buffer首节点在栈中除了栈顶节点以外，没有其他的孩子节点或者头节点，则将其进栈处理，执行"None-Shift"操作
        elif len(buffer) != 0 and not has_other_head_in_stack(
                b0) and not has_other_child_in_stack(b0):
            actions.append("NS")
            # Shift操作前，要将deque中暂存的节点先压栈
            while len(deque) != 0:
                stack.append(deque.pop())
            stack.append(buffer.pop())
            return

        # 栈顶节点没有未找到的孩子节点或头节点，说明完成了所有依存关系的生成，可以出栈丢弃了，执行"None-Reduce"操作
        elif s0 > 0 and not has_unfound_child(s0) and not lack_head(s0):
            actions.append("NR")
            stack.pop()
            return

        # 栈顶节点还有未找到的孩子节点或头节点，则将其暂时入deque保存，执行"None-Pass"操作
        elif s0 > 0:
            actions.append("NP")
            deque.append(stack.pop())
            return

        # 如果出现了意料之外的分支，那么就说明出错了
        else:
            actions.append('-E-')
            print('"error in oracle!"')
            return

    # 每次生成一步转移动作，终止条件为：buffer为空
    while len(buffer) != 0:
        get_oracle_actions_onestep()

    return actions


class TransitionField(Field):
    def __init__(self, *args, **kwargs):
        self.label_field = kwargs.pop(
            'label_field') if 'label_field' in kwargs else None
        self.transition_vocab = {}
        super().__init__(*args, **kwargs)

    def build(self, dataset, min_freq=1):
        charts = getattr(dataset, self.name)
        charts = [self.label_field.preprocess(chart) for chart in charts]
        charts = [[[
            self.label_field.vocab[i] if i is not None else -1 for i in row
        ] for row in chart] for chart in charts]
        all_transitions = []
        for chart in charts:
            # pdb.set_trace()
            annotated_sentence_len = len(chart)
            directed_arc_indices = []
            arc_tag_id = []
            for i in range(1, annotated_sentence_len):
                for j in range(annotated_sentence_len):
                    if (chart[i][j] != -1):
                        directed_arc_indices.append((i, j))
                        arc_tag_id.append(str(chart[i][j]))
            action = get_oracle_actions(annotated_sentence_len,
                                        directed_arc_indices, arc_tag_id)
            # 这里直接把边映射到编号
            all_transitions.append(action)

        self.counter = Counter(transition for transitions in all_transitions
                          for transition in transitions)
        self.vocab = Vocab(self.counter, min_freq, self.specials, self.unk_index)

    def transform(self, charts):
        charts = [self.label_field.preprocess(chart) for chart in charts]
        charts = [[[
            self.label_field.vocab[i] if i is not None else -1 for i in row
        ] for row in chart] for chart in charts]
        all_transitions = []
        for chart in charts:
            # pdb.set_trace()
            annotated_sentence_len = len(chart)
            directed_arc_indices = []
            arc_tag_id = []
            for i in range(1, annotated_sentence_len):
                for j in range(annotated_sentence_len):
                    if(chart[i][j] != -1):
                        directed_arc_indices.append((i, j))
                        arc_tag_id.append(str(chart[i][j]))
            action = get_oracle_actions(annotated_sentence_len,
                                        directed_arc_indices, arc_tag_id)
            # 这里直接把边映射到编号
            all_transitions.append(action)

        # counter = Counter(transition for transitions in all_transitions for transition in transitions)
        # self.vocab = Vocab(counter, 1, self.specials, self.unk_index)

        all_transitions = [[self.vocab[transition] for transition in transitions] for transitions in all_transitions]
        all_transitions = [torch.tensor(transitions) for transitions in all_transitions]
        return all_transitions

    @property
    def pad_index(self):
        if self.pad is None:
            return -1
        if hasattr(self, 'vocab'):
            return self.vocab[self.pad]
        return self.specials.index(self.pad)


class ParTransitionField(TransitionField):

    def build(self, dataset, min_freq=1):
        charts = getattr(dataset, self.name)
        charts = [self.label_field.preprocess(chart) for chart in charts]
        charts = [[[
            self.label_field.vocab[i] if i is not None else -1 for i in row
        ] for row in chart] for chart in charts]
        all_transitions = []
        for chart in charts:
            # pdb.set_trace()
            annotated_sentence_len = len(chart)
            directed_arc_indices = []
            arc_tag_id = []
            for i in range(1, annotated_sentence_len):
                for j in range(annotated_sentence_len):
                    if (chart[i][j] != -1):
                        directed_arc_indices.append((i, j))
                        # arc_tag_id.append(str(chart[i][j]))
                        arc_tag_id.append('-')

            action = get_oracle_actions(annotated_sentence_len,
                                        directed_arc_indices, arc_tag_id)
            all_transitions.append(action)

        self.counter = Counter(transition for transitions in all_transitions
                               for transition in transitions)
        self.vocab = Vocab(self.counter, min_freq, self.specials,
                           self.unk_index)

    def transform(self, charts):
        charts = [self.label_field.preprocess(chart) for chart in charts]
        charts = [[[
            self.label_field.vocab[i] if i is not None else -1 for i in row
        ] for row in chart] for chart in charts]
        all_transitions = []
        for chart in charts:
            # pdb.set_trace()
            annotated_sentence_len = len(chart)
            directed_arc_indices = []
            arc_tag_id = []
            for i in range(1, annotated_sentence_len):
                for j in range(annotated_sentence_len):
                    if (chart[i][j] != -1):
                        directed_arc_indices.append((i, j))
                        # arc_tag_id.append(str(chart[i][j]))
                        arc_tag_id.append('-')
            action = get_oracle_actions(annotated_sentence_len,
                                        directed_arc_indices, arc_tag_id)
            # 这里直接把边映射到编号
            all_transitions.append(action)

        # counter = Counter(transition for transitions in all_transitions for transition in transitions)
        # self.vocab = Vocab(counter, 1, self.specials, self.unk_index)

        all_transitions = [[
            self.vocab[transition] for transition in transitions
        ] for transitions in all_transitions]
        all_transitions = [
            torch.tensor(transitions) for transitions in all_transitions
        ]
        return all_transitions


class TranLabelField(Field):
    def __init__(self, *args, **kwargs):
        self.label_field = kwargs.pop(
            'label_field') if 'label_field' in kwargs else None
        self.transition_vocab = {}
        super().__init__(*args, **kwargs)

    def transform(self, charts):
        null_label_id = len(self.label_field.vocab)
        charts = [self.label_field.preprocess(chart) for chart in charts]
        charts = [[[
            self.label_field.vocab[i] if i is not None else -1 for i in row
        ] for row in chart] for chart in charts]
        all_transitions_label = []
        for chart in charts:
            # pdb.set_trace()
            annotated_sentence_len = len(chart)
            directed_arc_indices = []
            arc_tag_id = []
            for i in range(1, annotated_sentence_len):
                for j in range(annotated_sentence_len):
                    if (chart[i][j] != -1):
                        directed_arc_indices.append((i, j))
                        arc_tag_id.append(str(chart[i][j]))
            action = get_oracle_actions(annotated_sentence_len,
                                        directed_arc_indices, arc_tag_id)
            label = []
            for a in action:
                sp = a.split(':')
                if(len(sp) > 1):
                    label.append(int(sp[1]))
                else:
                    label.append(null_label_id)
            all_transitions_label.append(label)
        all_transitions_label = [
            torch.tensor(labels) for labels in all_transitions_label
        ]
        return all_transitions_label

    @property
    def pad_index(self):
        if self.pad is None:
            return -1
        if hasattr(self, 'vocab'):
            return self.vocab[self.pad]
        return self.specials.index(self.pad)



class ParStateField(Field):
    def __init__(self, *args, **kwargs):
        self.label_field = kwargs.pop(
            'label_field') if 'label_field' in kwargs else None
        self.transition_field = kwargs.pop(
            'transition_field') if 'transition_field' in kwargs else None
        # self.transition_vocab = {}
        super().__init__(*args, **kwargs)

    def transform(self, charts):
        charts = [self.label_field.preprocess(chart) for chart in charts]
        charts = [[[
            self.label_field.vocab[i] if i is not None else -1 for i in row
        ] for row in chart] for chart in charts]
        all_states = []
        for chart in charts:
            annotated_sentence_len = len(chart)
            directed_arc_indices = []
            arc_tag_id = []
            for i in range(1, annotated_sentence_len):
                for j in range(annotated_sentence_len):
                    if (chart[i][j] != -1):
                        directed_arc_indices.append((i, j))
                        # arc_tag_id.append(str(chart[i][j]))
                        arc_tag_id.append('-')
            states = get_oracle_states(annotated_sentence_len,
                                       directed_arc_indices, arc_tag_id)
            # 是否要加入label信息，目前之加入了action
            for state in states:
                actions = state[3]
                for i in range(len(actions)):
                    if (actions[i] != -1):
                        actions[i] = self.transition_field.vocab[actions[i]]
                state[3] = actions
            # pdb.set_trace()
            all_states.append(states)
        all_states = [torch.tensor(states) for states in all_states]
        return all_states

    @property
    def pad_index(self):
        if self.pad is None:
            return -1
        if hasattr(self, 'vocab'):
            return self.vocab[self.pad]
        return self.specials.index(self.pad)


class StateField(Field):
    def __init__(self, *args, **kwargs):
        self.label_field = kwargs.pop(
            'label_field') if 'label_field' in kwargs else None
        self.transition_field = kwargs.pop(
            'transition_field') if 'transition_field' in kwargs else None
        # self.transition_vocab = {}
        super().__init__(*args, **kwargs)

    def transform(self, charts):
        charts = [self.label_field.preprocess(chart) for chart in charts]
        charts = [[[
            self.label_field.vocab[i] if i is not None else -1 for i in row
        ] for row in chart] for chart in charts]
        all_states = []
        for chart in charts:
            annotated_sentence_len = len(chart)
            directed_arc_indices = []
            arc_tag_id = []
            for i in range(1, annotated_sentence_len):
                for j in range(annotated_sentence_len):
                    if (chart[i][j] != -1):
                        directed_arc_indices.append((i, j))
                        arc_tag_id.append(str(chart[i][j]))
            states = get_oracle_states(annotated_sentence_len, directed_arc_indices, arc_tag_id)
            for state in states:
                actions = state[3]
                for i in range(len(actions)):
                    if(actions[i] != -1):
                        actions[i] = self.transition_field.vocab[actions[i]]
                state[3] = actions
            # pdb.set_trace()
            all_states.append(states)
        all_states = [torch.tensor(states) for states in all_states]
        return all_states

    @property
    def pad_index(self):
        if self.pad is None:
            return -1
        if hasattr(self, 'vocab'):
            return self.vocab[self.pad]
        return self.specials.index(self.pad)
