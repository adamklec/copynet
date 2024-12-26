import torch
from torch import nn
from torch.autograd import Variable


class EncoderRNN(nn.Module):
    """
    实现了一个基于GRU（门控循环单元，Gated Recurrent Unit）的编码器模型.
    继承了 nn.Module，这是所有 PyTorch 模型的基类。通过继承它，EncoderRNN 可以利用 PyTorch 提供的各种功能（如自动求导、参数管理等）
    """
    def __init__(self, input_size, hidden_size, embedding_size):
        """
        params: input_size  输入词汇表大小（即词汇表的词语数目）
        hidden_size：GRU隐藏层的大小。
        embedding_size：词嵌入的维度（即将输入词语映射到的向量维度）。
        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(input_size, self.embedding_size)                          # 使用了 nn.Embedding 层来初始化一个词嵌入层。该层将输入的每个词 ID 映射到一个 embedding_size 维度的向量。
        self.embedding.weight.data.normal_(0, 1 / self.embedding_size**0.5)                     # 通过正态分布初始化嵌入层的权重，均值为 0，标准差为 \frac{1}{\sqrt{\text{embedding_size}}}，这种初始化方法常用于词嵌入层。
        self.gru = nn.GRU(embedding_size, hidden_size, bidirectional=True, batch_first=True)    # 定义了一个双向 GRU 层。它的输入维度是 embedding_size，输出维度是 hidden_size。bidirectional=True 表示该 GRU 是双向的，能够同时从前向和反向捕捉序列信息。batch_first=True 表示输入的张量格式为 (batch_size, seq_len, input_size)。

    def forward(self, iput, hidden, lengths):
        """
        params: iput 输入的词 ID 序列，形状为 (batch_size, seq_len)。
        params: hidden GRU的初始隐藏状态，形状为 (num_layers * num_directions, batch_size, hidden_size)。num_layers 是 GRU 的层数，num_directions 是 2（因为是双向的）。
        params: lengths 一个列表，表示每个输入序列的实际长度，用于处理变长序列。
        """
        # iput batch must be sorted by sequence length：
        # 这行代码用于处理超出词汇表的词 ID，将它们替换为 <UNK>（未知词）的 ID，通常是 3（<UNK> 的 ID）。这是为了防止输入的 ID 超出了词嵌入层的词汇表大小。
        iput = iput.masked_fill(iput > self.embedding.num_embeddings, 3)  # replace OOV words with <UNK> before embedding
        embedded = self.embedding(iput)                                                                     # 将输入的词 ID 序列通过词嵌入层转换成词向量序列。
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)      # 将嵌入后的序列打包为一个 PackedSequence 对象，它用于处理变长序列。lengths 参数指定了每个序列的实际长度，batch_first=True 表示输入格式是 (batch_size, seq_len, input_size)。
        self.gru.flatten_parameters()                                                                       # 优化 GRU 操作的性能，确保参数在不同设备（如 GPU）上的布局是连续的。
        output, hidden = self.gru(packed_embedded, hidden)                                                  # 将打包后的序列输入 GRU 层。output 是 GRU 的输出，hidden 是最后一个时刻的隐藏状态。
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)           # 将 PackedSequence 对象转换回一个常规的张量，并将变长序列填充成相同的长度。output_lengths 是每个序列的长度。
        return output, hidden

    def init_hidden(self, batch_size):
        """
        初始化隐藏状态。2 是因为 GRU 是双向的（即正向和反向各有一个隐藏状态）。batch_size 是批次大小，hidden_size 是每个方向的隐藏层大小。
        检查模型是否在 GPU 上运行。如果是，将隐藏状态转移到 GPU；否则，返回 CPU 上的隐藏状态。
        """
        hidden = Variable(torch.zeros(2, batch_size, self.hidden_size))  # bidirectional rnn
        if next(self.parameters()).is_cuda:
            return hidden.cuda()
        else:
            return hidden
