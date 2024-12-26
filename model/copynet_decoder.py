import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from dataset import Language
from utils import to_one_hot, DecoderBase


class CopyNetDecoder(DecoderBase):
    """
    具有“复制机制”和“生成机制”的解码器。这个类继承自 DecoderBase 类，用于处理序列到序列（Seq2Seq）任务
    """
    def __init__(self, hidden_size, embedding_size, lang: Language, max_length):
        """
        hidden_size 是隐藏层的大小，embedding_size 是词嵌入的维度，lang 是语言对象（通常包含词汇表），max_length 是生成的最大序列长度。
        """
        super(CopyNetDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.lang = lang
        self.max_length = max_length

        """
        定义了一个词嵌入层，词汇表大小是 len(self.lang.tok_to_idx)，嵌入维度是 embedding_size，padding_idx=0 表示 0 被用作填充标记的索引。
        使用正态分布初始化嵌入层权重，均值为 0，标准差为 \frac{1}{\sqrt{\text{embedding_size}}}。
        将嵌入矩阵的第一个词的权重（通常是填充符 <PAD>）设为全 0
        """
        self.embedding = nn.Embedding(len(self.lang.tok_to_idx), self.embedding_size, padding_idx=0)
        self.embedding.weight.data.normal_(0, 1 / self.embedding_size**0.5)
        self.embedding.weight.data[0, :] = 0.0

        """定义了两个全连接层（线性变换），分别用于注意力机制和复制机制。"""
        self.attn_W = nn.Linear(self.hidden_size, self.hidden_size)
        self.copy_W = nn.Linear(self.hidden_size, self.hidden_size)

        """
        定义了一个 GRU 层。输入维度是 2 * self.hidden_size + self.embedding.embedding_dim，即包含上下文向量、选择性读取向量和当前词的嵌入向量。
        定义了一个输出层，将 GRU 的输出映射到词汇表的大小，生成一个词概率分布。
        """
        self.gru = nn.GRU(2 * self.hidden_size + self.embedding.embedding_dim, self.hidden_size, batch_first=True)  # input = (context + selective read size + embedding)
        self.out = nn.Linear(self.hidden_size, len(self.lang.tok_to_idx))

    def forward(self, encoder_outputs, inputs, final_encoder_hidden, targets=None, keep_prob=1.0, teacher_forcing=0.0):
        """
        定义了如何通过解码器生成输出。它接收：
        encoder_outputs：来自编码器的输出，形状为 (batch_size, seq_length, hidden_size)。
        inputs：输入序列，用于初始化解码器的输入。
        final_encoder_hidden：编码器的最终隐藏状态。
        targets：目标序列（如果用于训练时的教师强制）。
        keep_prob：dropout 的概率。
        teacher_forcing：决定是否使用教师强制（即将目标作为输入，而不是使用模型的预测）。
        """
        batch_size = encoder_outputs.data.shape[0]
        seq_length = encoder_outputs.data.shape[1]
        """初始化解码器的隐藏状态，形状为 (1, batch_size, hidden_size)。如果模型在 GPU 上运行，隐藏状态会被转移到 GPU。"""
        hidden = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if next(self.parameters()).is_cuda:
            hidden = hidden.cuda()
        else:
            hidden = hidden

        # every decoder output seq starts with <SOS>
        """初始化起始标记 <SOS> 的输出张量，用于标记解码器开始生成序列。
        初始化每个时间步的预测标记。"""
        sos_output = Variable(torch.zeros((batch_size, self.embedding.num_embeddings + seq_length)))
        sampled_idx = Variable(torch.ones((batch_size, 1)).long())
        if next(self.parameters()).is_cuda:
            sos_output = sos_output.cuda()
            sampled_idx = sampled_idx.cuda()
        """将 <SOS> 标记的索引位置设置为 1，表示一热编码。"""
        sos_output[:, 1] = 1.0  # index 1 is the <SOS> token, one-hot encoding
        """
        用来存储解码器的输出，初始化时包含 <SOS>
        用来存储采样的词索引。"""
        decoder_outputs = [sos_output]
        sampled_idxs = [sampled_idx]

        if keep_prob < 1.0:
            dropout_mask = (Variable(torch.rand(batch_size, 1, 2 * self.hidden_size + self.embedding.embedding_dim)) < keep_prob).float() / keep_prob
        else:
            dropout_mask = None

        selective_read = Variable(torch.zeros(batch_size, 1, self.hidden_size))
        one_hot_input_seq = to_one_hot(inputs, len(self.lang.tok_to_idx) + seq_length)
        if next(self.parameters()).is_cuda:
            selective_read = selective_read.cuda()
            one_hot_input_seq = one_hot_input_seq.cuda()
        """逐步生成解码器输出，直到达到最大长度。"""
        for step_idx in range(1, self.max_length):

            if targets is not None and teacher_forcing > 0.0 and step_idx < targets.shape[1]:
                # replace some inputs with the targets (i.e. teacher forcing)
                teacher_forcing_mask = Variable((torch.rand((batch_size, 1)) < teacher_forcing), requires_grad=False)
                if next(self.parameters()).is_cuda:
                    teacher_forcing_mask = teacher_forcing_mask.cuda()
                sampled_idx = sampled_idx.masked_scatter(teacher_forcing_mask, targets[:, step_idx-1:step_idx])

            sampled_idx, output, hidden, selective_read = self.step(sampled_idx, hidden, encoder_outputs, selective_read, one_hot_input_seq, dropout_mask=dropout_mask)

            decoder_outputs.append(output)
            sampled_idxs.append(sampled_idx)

        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        sampled_idxs = torch.stack(sampled_idxs, dim=1)

        return decoder_outputs, sampled_idxs

    def step(self, prev_idx, prev_hidden, encoder_outputs, prev_selective_read, one_hot_input_seq, dropout_mask=None):
        """
        step 方法是解码器每一步生成的核心，涉及注意力机制、RNN（GRU）计算、复制机制和生成机制：
        """
        batch_size = encoder_outputs.shape[0]
        seq_length = encoder_outputs.shape[1]
        vocab_size = len(self.lang.tok_to_idx)

        # Attention mechanism
        transformed_hidden = self.attn_W(prev_hidden).view(batch_size, self.hidden_size, 1)
        attn_scores = torch.bmm(encoder_outputs, transformed_hidden)  # reduce encoder outputs and hidden to get scores. remove singleton dimension from multiplication.
        attn_weights = F.softmax(attn_scores, dim=1)  # apply softmax to scores to get normalized weights
        context = torch.bmm(torch.transpose(attn_weights, 1, 2), encoder_outputs)  # [b, 1, hidden] weighted sum of encoder_outputs (i.e. values)

        # Call the RNN
        out_of_vocab_mask = prev_idx > vocab_size  # [b, 1] bools indicating which seqs copied on the previous step
        unks = torch.ones_like(prev_idx).long() * 3
        prev_idx = prev_idx.masked_scatter(out_of_vocab_mask, unks)  # replace copied tokens with <UNK> token before embedding
        embedded = self.embedding(prev_idx)  # embed input (i.e. previous output token)

        rnn_input = torch.cat((context, prev_selective_read, embedded), dim=2)
        if dropout_mask is not None:
            if next(self.parameters()).is_cuda:
                dropout_mask = dropout_mask.cuda()
            rnn_input *= dropout_mask

        self.gru.flatten_parameters()
        output, hidden = self.gru(rnn_input, prev_hidden)  # state.shape = [b, 1, hidden]

        # Copy mechanism
        transformed_hidden2 = self.copy_W(output).view(batch_size, self.hidden_size, 1)
        copy_score_seq = torch.bmm(encoder_outputs, transformed_hidden2)  # this is linear. add activation function before multiplying.
        copy_scores = torch.bmm(torch.transpose(copy_score_seq, 1, 2), one_hot_input_seq).squeeze(1)  # [b, vocab_size + seq_length]
        missing_token_mask = (one_hot_input_seq.sum(dim=1) == 0)  # tokens not present in the input sequence
        missing_token_mask[:, 0] = 1  # <MSK> tokens are not part of any sequence
        copy_scores = copy_scores.masked_fill(missing_token_mask, -1000000.0)

        # Generate mechanism
        gen_scores = self.out(output.squeeze(1))  # [b, vocab_size]
        gen_scores[:, 0] = -1000000.0  # penalize <MSK> tokens in generate mode too

        # Combine results from copy and generate mechanisms
        combined_scores = torch.cat((gen_scores, copy_scores), dim=1)
        probs = F.softmax(combined_scores, dim=1)
        gen_probs = probs[:, :vocab_size]

        gen_padding = Variable(torch.zeros(batch_size, seq_length))
        if next(self.parameters()).is_cuda:
            gen_padding = gen_padding.cuda()
        gen_probs = torch.cat((gen_probs, gen_padding), dim=1)  # [b, vocab_size + seq_length]

        copy_probs = probs[:, vocab_size:]

        final_probs = gen_probs + copy_probs

        log_probs = torch.log(final_probs + 10**-10)

        _, topi = log_probs.topk(1)
        sampled_idx = topi.view(batch_size, 1)

        # Create selective read embedding for next time step
        reshaped_idxs = sampled_idx.view(-1, 1, 1).expand(one_hot_input_seq.size(0), one_hot_input_seq.size(1), 1)
        pos_in_input_of_sampled_token = one_hot_input_seq.gather(2, reshaped_idxs)  # [b, seq_length, 1]
        selected_scores = pos_in_input_of_sampled_token * copy_score_seq
        selected_scores_norm = F.normalize(selected_scores, p=1)

        selective_read = (selected_scores_norm * encoder_outputs).sum(dim=1).unsqueeze(1)

        return sampled_idx, log_probs, hidden, selective_read
