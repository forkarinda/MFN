import itertools
import math
import numpy as np
import pickle
import torch
import os

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

class Seq2SeqDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_dict, tgt_dict):
        self.src_dict, self.tgt_dict = src_dict, tgt_dict
        with open(src_file, 'rb') as f:
            self.src_dataset = pickle.load(f)
            self.src_sizes = np.array([len(tokens) for tokens in self.src_dataset])


        with open(tgt_file, 'rb') as f:
            self.tgt_dataset = pickle.load(f)
            self.tgt_sizes = np.array([len(tokens) for tokens in self.tgt_dataset])

    def __getitem__(self, index):
        return {
            'id': index,
            'source': torch.LongTensor(self.src_dataset[index]),
            'target': torch.LongTensor(self.tgt_dataset[index]),
        }

    def __len__(self):
        return len(self.src_dataset)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        if len(samples) == 0:
            return {}

        def merge(values, move_eos_to_beginning=False):
            max_length = max(v.size(0) for v in values)
            result = values[0].new(len(values), max_length).fill_(self.src_dict.pad_idx)
            for i, v in enumerate(values):
                if move_eos_to_beginning:
                    assert v[-1] == self.src_dict.eos_idx
                    result[i, 0] = self.src_dict.eos_idx
                    result[i, 1:len(v)] = v[:-1]
                else:
                    result[i, :len(v)].copy_(v)
            return result

        id = torch.LongTensor([s['id'] for s in samples])
        src_tokens = merge([s['source'] for s in samples])
        tgt_tokens = merge([s['target'] for s in samples])
        tgt_inputs = merge([s['target'] for s in samples], move_eos_to_beginning=True)
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])

  
       

        return {
            'id': id,
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'tgt_tokens': tgt_tokens,
            'tgt_inputs': tgt_inputs,
            'num_tokens': sum(len(s['target']) for s in samples),
        }

    
     

class How2Dataset(Dataset):
    def __init__(self, src_file, video_file, video_dir, tgt_file, all_dict):
        self.all_dict = all_dict
        with open(src_file, 'rb') as f:
            self.src_dataset = pickle.load(f)
            self.src_sizes = np.array([len(tokens) for tokens in self.src_dataset])
            #print(self.src_dataset[0],self.src_sizes[0])

        with open(tgt_file, 'rb') as f:
            self.tgt_dataset = pickle.load(f)
            self.tgt_sizes = np.array([len(tokens) for tokens in self.tgt_dataset])


        with open(video_file, 'r',encoding='utf8') as f:
            video_samples = [line.strip().split('/')[-1] for line in f]
        self.video_feature_path = [os.path.join(video_dir, index) for index in video_samples]

          
    def __getitem__(self, index):

        return {
            'id': index,
            'source': torch.LongTensor(self.src_dataset[index][:800]),
            'target': torch.LongTensor(self.tgt_dataset[index][:800]),
            'video': np.load(self.video_feature_path[index], encoding='bytes')[:1024]
        }

    def __len__(self):
        return len(self.src_dataset)

    def collater(self, samples):
       
        """Merge a list of samples to form a mini-batch."""
        if len(samples) == 0:
            return {}
        
        def merge_video(data):
            video_max_len = max(x.shape[0] for x in data)
            width = data[0].shape[1]
            video_padded = [np.zeros((video_max_len, width)) for _ in data] 
            for i,(pad, x) in enumerate(zip(video_padded, data)):
                video_padded[i][:x.shape[0]] = x
            video_padded = torch.tensor(video_padded).float()
            return video_padded
     
    
        def merge(values, move_eos_to_beginning=False):
            max_length = max(v.size(0) for v in values)
            result = values[0].new(len(values), max_length).fill_(self.all_dict.pad_idx)
            for i, v in enumerate(values):
                if move_eos_to_beginning:
                    assert v[-1] == self.all_dict.eos_idx
                    result[i, 0] = self.all_dict.eos_idx
                    result[i, 1:len(v)] = v[:-1]
                else:
                    result[i, :len(v)].copy_(v)
            return result

        id = torch.LongTensor([s['id'] for s in samples])
        src_tokens = merge([s['source'] for s in samples])
        tgt_tokens = merge([s['target'] for s in samples])
        tgt_inputs = merge([s['target'] for s in samples], move_eos_to_beginning=True)
        video_inputs = merge_video([s['video'] for s in samples])
    
        # Sort by descending source length
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])

        return {
            'id': id,
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'tgt_tokens': tgt_tokens,
            'tgt_inputs': tgt_inputs,
            'num_tokens': sum(len(s['target']) for s in samples),
            'video_inputs': video_inputs
            
        }

    
class BatchSampler(Sampler):
    def __init__(self, dataset, max_tokens=None, batch_size=None, num_shards=1, shard_id=0, shuffle=True, seed=42):
        self.dataset, self.shuffle, self.seed = dataset, shuffle, seed
        self.batch_size = batch_size if batch_size is not None else float('Inf')
        self.max_tokens = max_tokens if max_tokens is not None else float('Inf')
        self.batches = self._batch_generator()

        self.shard_len = int(math.ceil(len(self.batches) / num_shards))
        self.itr = itertools.zip_longest(
            range(self.shard_len),
            itertools.islice(self.batches, shard_id, len(self.batches), num_shards),
            fillvalue=[])
  
    def __len__(self):
        return self.shard_len

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.itr)[1]

    def _batch_generator(self):
        np.random.seed(self.seed)
        indices = np.random.permutation(len(self.dataset)) if self.shuffle else np.arange(len(self.dataset))
     

        batches, batch, sample_len = [], [], 0
        for idx in indices:
            batch.append(idx)
            sample_len = max(sample_len, self.dataset.tgt_sizes[idx])
            num_tokens = len(batch) * sample_len
            if len(batch) == self.batch_size or num_tokens > self.max_tokens:
                batches.append(batch)
                batch, sample_len = [], 0
        if len(batch) > 0:
            batches.append(batch)

        if self.shuffle:
            np.random.shuffle(batches)
        return batches
