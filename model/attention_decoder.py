import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from dataset import Language
from utils import DecoderBase


class AttentionDecoder(DecoderBase):
    def __init__(self, hidden_size, embedding_size, lang: Language, max_length):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.lang = lang
        self.max_length = max_length

        self.embedding = nn.Embedding(len(lang.tok_to_idx), self.embedding_size, padding_idx=0)
        self.embedding.weight.data.normal_(0, 1 / self.embedding_size**0.5)
        self.embedding.weight.data[0, :] = 0.0

        self.attn_W = nn.Linear(self.hidden_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size + self.embedding_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, len(lang.tok_to_idx))

    def forward(self, encoder_outputs, inputs, final_encoder_hidden, targets=None, keep_prob=1.0, teacher_forcing=0.0):
        batch_size = encoder_outputs.data.shape[0]

        hidden = Variable(torch.zeros(1, batch_size, self.hidden_size))  # overwrite the encoder hidden state with zeros
        if next(self.parameters()).is_cuda:
            hidden = hidden.cuda()
        else:
            hidden = hidden

        # every decoder output seq starts with <SOS>
        sos_output = Variable(torch.zeros((batch_size, self.embedding.num_embeddings)))
        sos_output[:, 1] = 1.0  # index 1 is the <SOS> token, one-hot encoding
        sos_idx = Variable(torch.ones((batch_size, 1)).long())

        if next(self.parameters()).is_cuda:
            sos_output = sos_output.cuda()
            sos_idx = sos_idx.cuda()

        decoder_outputs = [sos_output]
        sampled_idxs = [sos_idx]

        iput = sos_idx

        dropout_mask = torch.rand(batch_size, 1, self.hidden_size + self.embedding.embedding_dim)
        dropout_mask = dropout_mask <= keep_prob
        dropout_mask = Variable(dropout_mask).float() / keep_prob

        for step_idx in range(1, self.max_length):

            if targets is not None and teacher_forcing > 0.0:
                # replace some inputs with the targets (i.e. teacher forcing)
                teacher_forcing_mask = Variable((torch.rand((batch_size, 1)) <= teacher_forcing), requires_grad=False)
                if next(self.parameters()).is_cuda:
                    teacher_forcing_mask = teacher_forcing_mask.cuda()
                iput = iput.masked_scatter(teacher_forcing_mask, targets[:, step_idx-1:step_idx])

            output, hidden = self.step(iput, hidden, encoder_outputs, dropout_mask=dropout_mask)

            decoder_outputs.append(output)
            _, topi = decoder_outputs[-1].topk(1)
            iput = topi.view(batch_size, 1)
            sampled_idxs.append(iput)

        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        sampled_idxs = torch.stack(sampled_idxs, dim=1)

        return decoder_outputs, sampled_idxs

    def step(self, prev_idx, prev_hidden, encoder_outputs, dropout_mask=None):

        batch_size = prev_idx.shape[0]
        vocab_size = len(self.lang.tok_to_idx)

        # encoder_output * W * decoder_hidden for each encoder_output
        transformed_hidden = self.attn_W(prev_hidden).view(batch_size, self.hidden_size, 1)
        scores = torch.bmm(encoder_outputs, transformed_hidden).squeeze(2)  # reduce encoder outputs and hidden to get scores. remove singleton dimension from multiplication.
        attn_weights = F.softmax(scores, dim=1).unsqueeze(1)  # apply softmax to scores to get normalized weights
        context = torch.bmm(attn_weights, encoder_outputs)  # weighted sum of encoder_outputs (i.e. values)

        out_of_vocab_mask = prev_idx > vocab_size  # [b, 1] bools indicating which seqs copied on the previous step
        unks = torch.ones_like(prev_idx).long() * 3
        prev_idx = prev_idx.masked_scatter(out_of_vocab_mask, unks)  # replace copied tokens with <UNK> token before embedding

        embedded = self.embedding(prev_idx)  # embed input (i.e. previous output token)

        rnn_input = torch.cat((context, embedded), dim=2)
        if dropout_mask is not None:
            if next(self.parameters()).is_cuda:
                dropout_mask = dropout_mask.cuda()
            rnn_input *= dropout_mask

        output, hidden = self.gru(rnn_input, prev_hidden)

        output = self.out(output.squeeze(1))  # linear transformation to output size
        output = F.log_softmax(output, dim=1)  # log softmax non-linearity to convert to log probabilities

        return output, hidden

    def init_hidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if next(self.parameters()).is_cuda:
            return result.cuda()
        else:
            return result
