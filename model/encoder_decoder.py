from torch import nn
from .attention_decoder import AttentionDecoder
from .copynet_decoder import CopyNetDecoder
from utils import seq_to_string, tokens_to_seq, Chinese_English_Splitter
from spacy.lang.en import English
from spacy.lang.zh import Chinese
from .encoder import EncoderRNN
from torch.autograd import Variable


class EncoderDecoder(nn.Module):
    def __init__(self, lang, max_length, hidden_size, embedding_size, decoder_type):
        """
        lang 是一个语言对象，包含词汇表和词汇相关的映射。
        max_length 是最大生成序列的长度。
        hidden_size 是编码器和解码器的隐藏层大小。
        embedding_size 是嵌入层的维度。
        decoder_type 指定使用哪种解码器（注意力解码器或复制解码器）。
        """
        super(EncoderDecoder, self).__init__()

        self.lang = lang            # 核心是词表，双向映射字典
        # 7506
        self.encoder = EncoderRNN(len(self.lang.tok_to_idx),                # 初始化编码器 EncoderRNN，它的输入大小是词汇表的大小（len(self.lang.tok_to_idx)），hidden_size 和 embedding_size 是之前传入的参数。
                                  hidden_size,                              
                                  embedding_size)                           # 训练前指定的超参数
        self.decoder_type = decoder_type                                    # 存储传入的解码器类型（attn 或 copy）。
        decoder_hidden_size = 2 * self.encoder.hidden_size

        """根据 decoder_type 的值选择解码器类型。如果是 'attn'，则使用 AttentionDecoder，如果是 'copy'，则使用 CopyNetDecoder。"""
        if self.decoder_type == 'attn':                                     
            self.decoder = AttentionDecoder(decoder_hidden_size,
                                            embedding_size,
                                            lang,
                                            max_length)
        elif self.decoder_type == 'copy':
            self.decoder = CopyNetDecoder(decoder_hidden_size,
                                          embedding_size,
                                          lang,
                                          max_length)
        else:
            raise ValueError("decoder_type must be 'attn' or 'copy'")

    def forward(self, inputs, lengths, targets=None, keep_prob=1.0, teacher_forcing=0.0):
        """初始化编码器的隐藏状态（init_hidden），然后将输入传递给编码器，得到 encoder_outputs 和最终的隐藏状态。"""
        batch_size = inputs.data.shape[0]
        hidden = self.encoder.init_hidden(batch_size)                       # torch.Size([2, 1, 128])
        encoder_outputs, hidden = self.encoder(inputs, hidden, lengths)     # torch.Size([1, 13, 256])

        """将编码器的输出、输入和隐藏状态传递给解码器。根据是否启用教师强制（teacher_forcing），解码器会选择生成一个输出序列和相应的词索引。"""
        decoder_outputs, sampled_idxs = self.decoder(encoder_outputs,       # torch.Size([1, 200, 5017])
                                                     inputs,
                                                     hidden,
                                                     targets=targets,
                                                     teacher_forcing=teacher_forcing)
        return decoder_outputs, sampled_idxs

    def get_response(self, input_string):
        """用于将用户输入的文本转换为模型生成的响应。
        判断解码器是否为 CopyNetDecoder，从而确定是否使用扩展词汇表（即包含输入序列的词汇）。"""
        use_extended_vocab = isinstance(self.decoder, CopyNetDecoder)

        """如果模型没有 parser_ 属性，则创建一个 English 分词器，负责将输入的文本字符串转化为词令。"""
        if not hasattr(self, 'parser_'):
            self.parser_ = Chinese()
        """将输入字符串 input_string 进行分词处理，得到 input_tokens 列表。前后分别加上特殊的开始标记 <SOS> 和结束标记 <EOS>。"""
        idx_to_tok = self.lang.idx_to_tok
        tok_to_idx = self.lang.tok_to_idx

        '''原始代码
        input_tokens = self.parser_(' '.join(input_string.split()))     # 标准化文本输入很有用，确保输入的字符串没有多余的空格。在文本处理中，可能会遇到多个连续空格的情况，使用 split() 和 join() 可以确保字符串格式统一。
        input_tokens = ['<SOS>'] + [token.orth_.lower() for token in input_tokens] + ['<EOS>']      # 将 input_tokens 中的每个 Token 的文本（orth_）转换为小写。'''

        input_tokens = Chinese_English_Splitter( input_string, part="nl" )
        input_tokens = ['<SOS>'] + [token.lower() for token in input_tokens] + ['<EOS>']
        
        """将 input_tokens 转换为索引序列 input_seq，并将其封装为一个 Variable（PyTorch 的张量对象）。"""
        input_seq = tokens_to_seq(input_tokens, tok_to_idx, len(input_tokens), use_extended_vocab)      # tensor([[   1,   74,   32, 1524,    7,    4,  255,   11,  443,   42, 5014,  254, 2]])
        input_variable = Variable(input_seq).view(1, -1)            # shape = torch.Size([1, 13])

        """如果模型在 GPU 上运行，将 input_variable 移动到 GPU。"""
        if next(self.parameters()).is_cuda:
            input_variable = input_variable.cuda()
        """调用 forward 方法，得到解码器的输出和生成的词索引。"""
        outputs, idxs = self.forward(input_variable, [len(input_seq)])      # torch.Size([1, 200, 5017]), torch.Size([1, 200, 1])
        """将 idxs（词索引）转化为一维向量。
        找到 <EOS> 标记的索引 eos_idx，然后将词索引转化为对应的字符串 output_string。"""
        idxs = idxs.data.view(-1)
        eos_idx = list(idxs).index(2) if 2 in list(idxs) else len(idxs)
        output_string = seq_to_string(idxs[:eos_idx + 1], idx_to_tok, input_tokens=input_tokens)

        return output_string

    def interactive(self, unsmear):
        """用于与模型进行交互。在控制台中输入消息，模型将生成响应。"""
        while True:
            input_string = input("\nType a message to Amy:\n")
            output_string = self.get_response(input_string)

            if unsmear:
                output_string = output_string.replace('<SOS>', '')
                output_string = output_string.replace('<EOS>', '')

            print('\nAmy:\n', output_string)
