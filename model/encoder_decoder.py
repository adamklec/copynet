from torch import nn
from .attention_decoder import AttentionDecoder
from .copynet_decoder import CopyNetDecoder
from utils import seq_to_string, tokens_to_seq
from spacy.lang.en import English
from .encoder import EncoderRNN
from torch.autograd import Variable


class EncoderDecoder(nn.Module):
    def __init__(self, lang, max_length, hidden_size, embedding_size, decoder_type):
        super(EncoderDecoder, self).__init__()

        self.lang = lang

        self.encoder = EncoderRNN(len(self.lang.tok_to_idx),
                                  hidden_size,
                                  embedding_size)
        self.decoder_type = decoder_type
        decoder_hidden_size = 2 * self.encoder.hidden_size
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

        batch_size = inputs.data.shape[0]
        hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs, hidden = self.encoder(inputs, hidden, lengths)
        decoder_outputs, sampled_idxs = self.decoder(encoder_outputs,
                                                     inputs,
                                                     hidden,
                                                     targets=targets,
                                                     teacher_forcing=teacher_forcing)
        return decoder_outputs, sampled_idxs

    def get_response(self, input_string):
        use_extended_vocab = isinstance(self.decoder, CopyNetDecoder)

        if not hasattr(self, 'parser_'):
            self.parser_ = English()

        idx_to_tok = self.lang.idx_to_tok
        tok_to_idx = self.lang.tok_to_idx

        input_tokens = self.parser_(' '.join(input_string.split()))
        input_tokens = ['<SOS>'] + [token.orth_.lower() for token in input_tokens] + ['<EOS>']
        input_seq = tokens_to_seq(input_tokens, tok_to_idx, len(input_tokens), use_extended_vocab)
        input_variable = Variable(input_seq).view(1, -1)

        if next(self.parameters()).is_cuda:
            input_variable = input_variable.cuda()

        outputs, idxs = self.forward(input_variable, [len(input_seq)])
        idxs = idxs.data.view(-1)
        eos_idx = list(idxs).index(2) if 2 in list(idxs) else len(idxs)
        output_string = seq_to_string(idxs[:eos_idx + 1], idx_to_tok, input_tokens=input_tokens)

        return output_string

    def interactive(self, unsmear):
        while True:
            input_string = input("\nType a message to Amy:\n")
            output_string = self.get_response(input_string)

            if unsmear:
                output_string = output_string.replace('<SOS>', '')
                output_string = output_string.replace('<EOS>', '')

            print('\nAmy:\n', output_string)
