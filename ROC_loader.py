import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import os
import pickle
import nltk
import json
from PIL import Image
from PIL import ImageFile
from build_vocab import Vocabulary
import operator
from functools import reduce
from transformer import Constants

class ROCDataset(Dataset):
    def __init__(self,
                 story_vocab,
                 frame_vocab,
                 text_path):
        self.dialogs = json.load(open(text_path, 'r'))
        self.max_sentence_len = 24
        self.story_vocab = story_vocab
        self.frame_vocab = frame_vocab

    def __getitem__(self, index):
        frame = []
        story = []

        dialog = self.dialogs[str(index)]
        for i, sen in enumerate(dialog):
            sentence = []
            tmp_frame = []
            Frame = sen['mapped_seq']
            description = sen['ner_description']
            tokens = description.strip().split()

            sentence.extend([self.story_vocab(token) for token in tokens])
            tmp_frame.extend([self.frame_vocab(F) for F in Frame if self.frame_vocab(F)!=Constants.UNK])
            if len(sentence) > self.max_sentence_len-1:
                sentence = sentence[:self.max_sentence_len]
            if len(tmp_frame) > self.max_sentence_len-1:
                tmp_frame = tmp_frame[:self.max_sentence_len]

            frame.append(tmp_frame)
            story.append(sentence)

        S, F, F_sen_pos = [], [], []
        for i, (s, f ) in enumerate(zip(story, frame)):
            S.append(Constants.BOSs[i])
            F.append(Constants.BOSs[i])
            S.extend(s)
            F.extend(f)
            F_sen_pos.extend([i]*(len(f)+1))
        #print(len(F), len(F_sen_pos))
        assert len(F) == len(F_sen_pos)
        return S, F, F_sen_pos

    def __len__(self):
        return len(self.dialogs)

def ROC_collate_fn(data):

    #List of sentences and frames [B,]
    stories, frames, f_sen_pos = zip(*data)

    lengths = [len(x)+1 for x in stories]
    #max_seq_len = max(lengths)
    max_seq_len = 126
    pad_stories = [s[:max_seq_len]+ [Constants.EOS] + [Constants.PAD for _ in range(max_seq_len - len(s) - 1)] for s in stories]
    stories_pos = [[pos_i+1 if w_i != 0 else 0
         for pos_i, w_i in enumerate(inst)] for inst in pad_stories]


    frame_lengths = [len(x)+1 for x in frames]
    #max_frame_seq_len = max(frame_lengths)
    max_frame_seq_len = 126
    pad_frame = [s + [Constants.EOS] + [Constants.PAD for _ in range(max_frame_seq_len - len(s) - 1)] for s in frames]
    pad_f_sen_pos = [s + [Constants.PAD for _ in range(max_frame_seq_len - len(s))] for s in f_sen_pos]
    frame_pos = [[pos_i+1 if w_i != 0 else 0
         for pos_i, w_i in enumerate(inst)] for inst in pad_frame]

    # print("texts",texts)
    # print("frame",frame)
    # print("lengths",max_seq_len)
    # print("max_frame_seq_len",max_frame_seq_len)
    targets = torch.LongTensor(pad_stories).view(-1, max_seq_len)
    lengths = torch.LongTensor(lengths).view(-1,1)
    targets_pos = torch.LongTensor(stories_pos).view(-1, max_seq_len)
    frame = torch.LongTensor(pad_frame).view(-1,max_frame_seq_len)
    frame_lengths = torch.LongTensor(frame_lengths).view(-1,1)
    frame_pos = torch.LongTensor(frame_pos).view(-1, max_frame_seq_len)
    frame_sen_pos = torch.LongTensor(pad_f_sen_pos).view(-1, max_frame_seq_len)

    return frame, frame_pos, frame_sen_pos, targets, targets_pos



def get_ROC_loader(text, roc_vocab, frame_vocab, batch_size, shuffle, num_workers, fixed_len=False, is_flat = False):
    ROC = ROCDataset(roc_vocab,
                     frame_vocab,
                     text_path=text)

    data_loader = torch.utils.data.DataLoader(dataset=ROC,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=ROC_collate_fn)
    return data_loader
