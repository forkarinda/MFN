import argparse
import collections
import logging
import os
import re
import pickle

from seq2seq import utils
from seq2seq.data.dictionary import Dictionary

SPACE_NORMALIZER = re.compile("\s+")


def word_tokenize(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def get_args():
    parser = argparse.ArgumentParser('Data pre-processing)')
  
    parser.add_argument('--train-prefix', default='../how2data/text/sum_train', help='train file prefix')
    parser.add_argument('--valid-prefix', default='../how2data/text/sum_cv', help='valid file prefix')
    parser.add_argument('--test-prefix', default='../how2data/text/sum_devtest',  help='test file prefix')
    parser.add_argument('--dest-dir', default='data-bin/how2', metavar='DIR', help='destination dir')

    parser.add_argument('--threshold', default=0, type=int, help='map words appearing less than threshold times to unknown')
    parser.add_argument('--num-words', default=-1, type=int, help='number of source words to retain')

    return parser.parse_args()


def main(args):
    os.makedirs(args.dest_dir, exist_ok=True)
#     all_dict = build_dictionary(['../how2-dataset/how/text/sum_asr_text.txt'])
   
#     all_dict.finalize(threshold=args.threshold, num_words=args.num_words)
#     all_dict.save(os.path.join(args.dest_dir, 'dict.all'))
#     logging.info('Built a dictionary with {} words'.format(len(all_dict)))
    all_dict = Dictionary.load(os.path.join(args.dest_dir, 'dict.{}'.format('all')))
    logging.info('Loaded a source dictionary with {} words'.format(len(all_dict)))
   
    def make_split_datasets(dictionary):
        if args.train_prefix is not None:
            make_binary_dataset(args.train_prefix + '/tr_tran_text.txt', os.path.join(args.dest_dir, 'train.' + 'tran'), dictionary)
            make_binary_dataset(args.train_prefix + '/tr_desc_text.txt', os.path.join(args.dest_dir, 'train.' + 'desc'), dictionary)
        if args.valid_prefix is not None:
            make_binary_dataset(args.valid_prefix + '/cv_tran_text.txt', os.path.join(args.dest_dir, 'valid.' + 'tran'), dictionary)
            make_binary_dataset(args.valid_prefix + '/cv_desc_text.txt', os.path.join(args.dest_dir, 'valid.' + 'desc'), dictionary)
        if args.test_prefix is not None:
            make_binary_dataset(args.test_prefix + '/dete_tran_text.txt', os.path.join(args.dest_dir, 'test.' + 'tran'), dictionary)
            make_binary_dataset(args.test_prefix + '/dete_desc_text.txt', os.path.join(args.dest_dir, 'test.' + 'desc'), dictionary)

    make_split_datasets(all_dict)



def build_dictionary(filenames, tokenize=word_tokenize):
    dictionary = Dictionary()
    for filename in filenames:
        with open(filename, 'r') as file:
            for line in file:
                for symbol in word_tokenize(line.strip()):
                    dictionary.add_word(symbol)
                dictionary.add_word(dictionary.eos_word)
    return dictionary


def make_binary_dataset(input_file, output_file, dictionary, tokenize=word_tokenize, append_eos=True):
    nsent, ntok = 0, 0
    unk_counter = collections.Counter()
    def unk_consumer(word, idx):
        if idx == dictionary.unk_idx and word != dictionary.unk_word:
            unk_counter.update([word])

    tokens_list = []
    with open(input_file, 'r') as inf:
        for line in inf:
            tokens = dictionary.binarize(line.strip(), word_tokenize, append_eos, consumer=unk_consumer)
            nsent, ntok = nsent + 1, ntok + len(tokens)
            tokens_list.append(tokens.numpy())

   
    with open(output_file, 'wb') as outf:
        pickle.dump(tokens_list, outf, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info('Built a binary dataset for {}: {} sentences, {} tokens, {:.3f}% replaced by unknown token'.format(
            input_file, nsent, ntok, 100.0 * sum(unk_counter.values()) / ntok, dictionary.unk_word))


if __name__ == '__main__':
    args = get_args()
    utils.init_logging(args)
    main(args)
