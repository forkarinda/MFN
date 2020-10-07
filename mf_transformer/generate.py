import argparse
import logging
import os
import torch

from preprocess import word_tokenize

from seq2seq import models, utils
from seq2seq.data.dataset import Seq2SeqDataset, BatchSampler, How2Dataset
from seq2seq.data.dictionary import Dictionary
from seq2seq.generator import SequenceGenerator

from tqdm import tqdm
from torch.serialization import default_restore_location
from termcolor import colored


def get_args():
    parser = argparse.ArgumentParser('Sequence to Sequence Model')
    parser.add_argument('--seed', default=42, type=int, help='pseudo random number generator seed')

    # Add data arguments
    parser.add_argument('--data', default='data-bin/how2', help='path to data directory')
    parser.add_argument('--video_file', default='../how2data/text/sum_devtest/dete_action.txt', help='name of video file')
    parser.add_argument('--video_dir', default='../how2data/video_action_features', help='path of video features')
    parser.add_argument('--checkpoint-path', default='checkpoints/mfn/checkpoint50_4.102.pt', help='path to the model file')
    parser.add_argument('--max-tokens', default=16000, type=int, help='maximum number of tokens in a batch')
    parser.add_argument('--batch-size', default=1, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--num-workers', default=2, type=int, help='number of data workers')

    parser.add_argument('--beam-size', default=6, type=int, help='beam size')
    parser.add_argument('--max-len', default=200, type=int, help='maximum length of generated sequence')
    parser.add_argument('--stop-early', default='True', help='stop generation immediately after finalizing hypotheses')
    parser.add_argument('--normalize_scores', default='True', help='normalize scores by the length of the output')
    parser.add_argument('--len-penalty', default=1, type=float, help='length penalty: > 1.0 favors longer sentences')
    parser.add_argument('--unk-penalty', default=0, type=float, help='unknown word penalty: >0 produces fewer unks')
    parser.add_argument('--remove-bpe', default='@@ ', help='remove BPE tokens before scoring')
    parser.add_argument('--num-hypo', default=1, type=int, help='number of hypotheses to output')
    parser.add_argument('--quiet', action='store_true', help='only print final scores')
    
    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # Load arguments from checkpoint
    torch.manual_seed(args.seed)
    state_dict = torch.load(args.checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    args = argparse.Namespace(**{**vars(args), **vars(state_dict['args'])})
    utils.init_logging(args)

    # Load dictionaries
    all_dict = Dictionary.load(os.path.join(args.data, 'dict.{}'.format('all')))
    logging.info('Loaded a source dictionary with {} words'.format(len(all_dict)))

    # Load dataset
    test_dataset = How2Dataset(
              src_file=os.path.join(args.data, 'test.{}'.format('tran')),
              tgt_file=os.path.join(args.data, 'test.{}'.format('desc')),
              all_dict=all_dict,
              video_file=args.video_file, 
              video_dir=args.video_dir)  

    test_loader = torch.utils.data.DataLoader(
        test_dataset, num_workers=args.num_workers, collate_fn=test_dataset.collater,
        batch_sampler=BatchSampler(
            test_dataset, args.max_tokens, 1,1,#args.batch_size,args.distributed_world_size,
            args.distributed_rank, shuffle=False, seed=args.seed))
   

    
    # Build model and criterion
    model = models.build_model(args, all_dict).cuda()

    model.load_state_dict(state_dict['model'])
    logging.info('Loaded a model from checkpoint {}'.format(args.checkpoint_path))

    translator = SequenceGenerator(
        model, all_dict, beam_size=args.beam_size, maxlen=args.max_len, stop_early=eval(args.stop_early),
        normalize_scores=eval(args.normalize_scores), len_penalty=args.len_penalty, unk_penalty=args.unk_penalty,
    )

    progress_bar = tqdm(test_loader, desc='| Generation', leave=False)
    for i, sample in enumerate(progress_bar):
        #logging.info(sample)
        sample = utils.move_to_cuda(sample)

        with torch.no_grad():
            hypos = translator.generate(sample['src_tokens'], sample['src_lengths'], sample['video_inputs'])
        for i, (sample_id, hypos) in enumerate(zip(sample['id'].data, hypos)):
            src_tokens = utils.strip_pad(sample['src_tokens'].data[i, :], all_dict.pad_idx)
            has_target = sample['tgt_tokens'] is not None
            target_tokens = utils.strip_pad(sample['tgt_tokens'].data[i, :], all_dict.pad_idx).int().cpu() if has_target else None

            src_str = all_dict.string(src_tokens, args.remove_bpe)
            target_str = all_dict.string(target_tokens, args.remove_bpe) if has_target else ''

            if not args.quiet:
                print('S-{}\t{}'.format(sample_id, src_str))
                if has_target:
                    print('T-{}\t{}'.format(sample_id, colored(target_str, 'green')))

            # Process top predictions
            for i, hypo in enumerate(hypos[:min(len(hypos), args.num_hypo)]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'].int().cpu(),
                    tgt_dict=all_dict,
                    remove_bpe=args.remove_bpe,
                )

                if not args.quiet:
                    print('H-{}\t{}'.format(sample_id, colored(hypo_str, 'blue')))
                    print('P-{}\t{}'.format(sample_id, ' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))))
                    print('A-{}\t{}'.format(sample_id, ' '.join(map(lambda x: str(x.item()), alignment))))

                # Score only the top hypothesis
                if has_target and i == 0:
                    # Convert back to tokens for evaluation with unk replacement and/or without BPE
                    target_tokens = all_dict.binarize(target_str, word_tokenize, add_if_not_exist=True)
                    print(target_tokens)

if __name__ == '__main__':
    args = get_args()
    main(args)
