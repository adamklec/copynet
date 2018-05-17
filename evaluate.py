import argparse
import random

import torch
from torch.autograd import Variable

from dataset import SequencePairDataset
from utils import seq_to_string, to_np, trim_seqs
from model.encoder_decoder import EncoderDecoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def evaluate(encoder_decoder: EncoderDecoder, data_loader):

    loss_function = torch.nn.NLLLoss(ignore_index=0, reduce=False) # what does this return for ignored idxs? same length output?

    losses = []
    all_output_seqs = []
    all_target_seqs = []

    for batch_idx, (input_idxs, target_idxs, _, _) in enumerate(tqdm(data_loader)):
        input_lengths = (input_idxs != 0).long().sum(dim=1)

        sorted_lengths, order = torch.sort(input_lengths, descending=True)

        input_variable = Variable(input_idxs[order, :][:, :max(input_lengths)], volatile=True)
        target_variable = Variable(target_idxs[order, :], volatile=True)
        batch_size = input_variable.shape[0]

        output_log_probs, output_seqs = encoder_decoder(input_variable, list(sorted_lengths))
        all_output_seqs.extend(trim_seqs(output_seqs))
        all_target_seqs.extend([list(seq[seq > 0])] for seq in to_np(target_variable))

        flattened_log_probs = output_log_probs.view(batch_size * encoder_decoder.decoder.max_length, -1)
        batch_losses = loss_function(flattened_log_probs, target_variable.contiguous().view(-1))
        losses.extend(list(to_np(batch_losses)))

    mean_loss = len(losses) / sum(losses)

    bleu_score = corpus_bleu(all_target_seqs, all_output_seqs, smoothing_function=SmoothingFunction().method1)

    return mean_loss, bleu_score


def print_output(input_seq, encoder_decoder: EncoderDecoder, input_tokens=None, target_tokens=None, target_seq=None):
    idx_to_tok = encoder_decoder.lang.idx_to_tok
    if input_tokens is not None:
        input_string = ' '.join(input_tokens)
    else:
        input_string = seq_to_string(input_seq, idx_to_tok)

    lengths = list((input_seq != 0).long().sum(dim=0))
    input_variable = Variable(input_seq).view(1, -1)
    target_variable = Variable(target_seq).view(1, -1)

    if target_tokens is not None:
        target_string = ' '.join(target_tokens)
    elif target_seq is not None:
        target_string = seq_to_string(target_seq, idx_to_tok, input_tokens=input_tokens)
    else:
        target_string = ''

    if target_seq is not None:
        target_eos_idx = list(target_seq).index(2) if 2 in list(target_seq) else len(target_seq)
        target_outputs, _ = encoder_decoder(input_variable, lengths, targets=target_variable, teacher_forcing=1.0)
        target_log_prob = sum([target_outputs[0, step_idx, target_idx] for step_idx, target_idx in enumerate(target_seq[:target_eos_idx+1])])

    outputs, idxs = encoder_decoder(input_variable, lengths)
    idxs = idxs.data.view(-1)
    eos_idx = list(idxs).index(2) if 2 in list(idxs) else len(idxs)
    string = seq_to_string(idxs[:eos_idx+1], idx_to_tok, input_tokens=input_tokens)
    log_prob = sum([outputs[0, step_idx, idx] for step_idx, idx in enumerate(idxs[:eos_idx+1])])

    print('>', input_string, '\n',flush=True)

    if target_seq is not None:
        print('=', target_string, flush=True)
    print('<', string, flush=True)

    print('\n')

    if target_seq is not None:
        print('target log prob:', float(target_log_prob))
    print('output log prob:', float(log_prob))

    print('-' * 100, '\n')

    return idxs


def main(model_name, use_cuda, n_print, idxs_print, use_train_dataset, val_size, batch_size, interact, unsmear):
    model_path = './model/' + model_name + '/'

    if use_cuda:
        encoder_decoder = torch.load(model_path + model_name + '.pt')
    else:
        encoder_decoder = torch.load(model_path + model_name + '.pt', map_location=lambda storage, loc: storage)

    if use_cuda:
        encoder_decoder = encoder_decoder.cuda()
    else:
        encoder_decoder = encoder_decoder.cpu()

    dataset = SequencePairDataset(lang=encoder_decoder.lang,
                                  use_cuda=use_cuda,
                                  is_val=not use_train_dataset,
                                  val_size=val_size)

    data_loader = DataLoader(dataset, batch_size=batch_size)
    if interact:
        encoder_decoder.interactive(unsmear)

    if n_print is not None:
        for _ in range(n_print):
            i_seq, t_seq, i_str, t_str = random.choice(dataset)

            i_length = (i_seq > 0).sum()
            t_length = (i_seq > 0).sum()

            i_seq = i_seq[:i_length]
            t_seq = t_seq[:t_length]

            i_tokens = i_str.split()
            t_tokens = t_str.split()

            print_output(i_seq, encoder_decoder, input_tokens=i_tokens, target_tokens=t_tokens, target_seq=t_seq)

    elif idxs_print is not None:
        for idx in idxs_print:
            i_seq, t_seq, i_str, t_str = dataset[idx]

            i_length = (i_seq > 0).sum()
            t_length = (i_seq > 0).sum()

            i_seq = i_seq[:i_length]
            t_seq = t_seq[:t_length]

            i_tokens = i_str.split()[:i_length]
            t_tokens = t_str.split()

            print_output(i_seq, encoder_decoder, input_tokens=i_tokens, target_tokens=t_tokens, target_seq=t_seq)

    else:
        evaluate(encoder_decoder, data_loader)


if __name__ == '__main__':
    random = random.Random(42)  # https://groups.google.com/forum/#!topic/nzpug/o4OW1O_4rgw

    arg_parser = argparse.ArgumentParser(description='Parse training parameters')
    arg_parser.add_argument('model_name', type=str,
                            help='The name of a subdirectory of ./model/ that '
                             'contains encoder and decoder model files.')

    arg_parser.add_argument('--n_print', type=int,
                            help='Instead of evaluating the model on the entire dataset,'
                             'n random examples from the dataset will be transformed.'
                             'The output will be printed.')

    arg_parser.add_argument('--idxs_print', nargs='+', type=int,
                            help='Instead of evaluating the model on the entire dataset,'
                             'the integers in this list will be used to select specific examples'
                             'to transform. The output will be printed.')

    arg_parser.add_argument('--interact', action='store_true',
                            help='Take model inputs from the keyboard.')

    arg_parser.add_argument('--use_cuda', action='store_true',
                            help='A flag indicating that cuda will be used.')

    arg_parser.add_argument('--use_train_dataset', action='store_true',
                            help='A flag that examples from the training dataset will be used as inputs.')

    arg_parser.add_argument('--val_size', type=float, default=0.1,
                            help='The fractional size of the validation split.')

    arg_parser.add_argument('--batch_size', type=int, default=100,
                            help='The batch size to use when evaluating on the full dataset.')

    arg_parser.add_argument('--unsmear', action='store_true',
                            help='Replace <NUM> tokens with "1" and remove <SOS> and <EOS> tokens.')

    args = arg_parser.parse_args()

    try:
        main(args.model_name,
             args.use_cuda,
             args.n_print,
             args.idxs_print,
             args.use_train_dataset,
             args.val_size,
             args.batch_size,
             args.interact,
             args.unsmear)
    except KeyboardInterrupt:
        pass
