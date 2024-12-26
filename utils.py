import numpy as np
import torch
from torch.autograd import Variable
from abc import ABC
from torch import nn
import matplotlib.pyplot as plt
import os
import jieba

def Chinese_English_Splitter( text, part = "query"):
    # 使用 jieba 进行分词
    if part == "query":
        """query语句保留空格"""
        sub_text = text.split(" ")              # 先保留query语句中原有的空格
        sub_split = []
        for sub_t in sub_text:                  # 再对每个没有空格的子段进行分隔
            if sub_t.strip() != "":              # 避免多个空格的情况
                tokens = jieba.cut(sub_t)
                ret_token = [ tok for tok in tokens if tok.strip()!= ""]
                sub_split.append( ret_token )

        ret_list = []
        for i in range( len(sub_split)):        # 子段extend，空格append
            ret_list.extend( sub_split[i] )
            if i != len(sub_split )-1:
                ret_list.append(" ")
        return ret_list
    elif part == "nl":
        """自然语言查询不要空格"""
        tokens = jieba.cut(text)
        ret_token = [ tok for tok in tokens if tok.strip()!= ""]
        return ret_token


# 创建图形和第一个坐标轴
def plots( val_loss, val_bleu, root_path ):
    epochs = [1+i for i in range(len(val_loss))]
    fig, ax1 = plt.subplots()

    # 绘制 val loss 曲线
    ax1.plot(epochs, val_loss, color='tab:red', label='val loss', marker='o')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('val loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # 创建第二个坐标轴，共享 x 轴
    ax2 = ax1.twinx()

    # 绘制 val BLEU 曲线
    ax2.plot(epochs, val_bleu, color='tab:blue', label='val BLEU', marker='s')
    ax2.set_ylabel('val BLEU', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # 设置标题
    plt.title('Training Progress: val loss vs val BLEU')

    # 显示图例
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.savefig( os.path.join( root_path,'training_progress.png' ), dpi=300)
    # 显示图形
    plt.show()

class DecoderBase(ABC, nn.Module):
    def forward(self, encoder_outputs, inputs, final_encoder_hidden, targets=None, teacher_forcing=1.0):
        raise NotImplementedError


def to_np(x):
    return x.data.cpu().numpy()


def seq_to_string(seq, idx_to_tok, input_tokens=None):
    vocab_size = len(idx_to_tok)
    seq_length = (seq != 0).sum()
    words = []
    for idx in seq[:seq_length]:
        if idx < vocab_size:
            words.append(idx_to_tok[int(idx)])
        elif input_tokens is not None:
            words.append(input_tokens[idx - vocab_size])
        else:
            words.append('<???>')
    string = ''.join(words)             # 将空格也作为学习的一部分
    return string


def largest_indices(ary, n):
    """Returns the indicies of the n largest values in a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).contiguous().view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


def tokens_to_seq(tokens, tok_to_idx, max_length, use_extended_vocab, input_tokens=None, ):
    seq = torch.zeros(max_length).long()
    tok_to_idx_extension = dict()

    for pos, token in enumerate(tokens):
        if token in tok_to_idx:
            idx = tok_to_idx[token]

        elif token in tok_to_idx_extension:
            idx = tok_to_idx_extension[token]

        elif use_extended_vocab and input_tokens is not None:
            # If the token is not in the vocab and an input token sequence was provided
            # find the position of the first occurance of the token in the input sequence
            # the token index in the output sequence is size of the vocab plus the position in the input sequence.
            # If the token cannot be found in the input sequence use the unknown token.

            tok_to_idx_extension[token] = tok_to_idx_extension.get(token,
                                 next((pos + len(tok_to_idx)
                                       for pos, input_token in enumerate(input_tokens)
                                       if input_token == token), 3))
            idx = tok_to_idx_extension[token]

        elif use_extended_vocab:
            # unknown tokens in the input sequence use the position of the first occurence + vocab_size as their index
            idx = pos + len(tok_to_idx)
        else:
            idx = tok_to_idx['<UNK>']

        seq[pos] = idx

    return seq


def trim_seqs(seqs):
    trimmed_seqs = []
    for output_seq in seqs:
        trimmed_seq = []
        for idx in to_np(output_seq):
            trimmed_seq.append(idx[0])
            if idx == 2:
                break
        trimmed_seqs.append(trimmed_seq)
    return trimmed_seqs


def get_seq_lengths(seqs):
    lengths = []
    for seq in seqs:
        seq = list(to_np(seq))
        lengths.append(seq.index(2) + 1 if 2 in seq else len(seq))
    return lengths


def contains_digit(string):
    return any(char.isdigit() for char in string)
