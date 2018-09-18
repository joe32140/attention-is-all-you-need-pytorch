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


class ROCDataset(Dataset):
    def __init__(self,
                 story_vocab,
                 frame_vocab,
                 text_path= None):
        self.dialogs = json.load(open(text_path, 'r'))

        self.story_vocab = story_vocab
        self.frame_vocab = frame_vocab

    def __getitem__(self, index):
        index = str(index)
        dialog = self.dialogs[index]
        story = []
        Frame = dialog['mapped_seq']
        # LU = dialog['LU']
        description = dialog['ner_description']

        # print("description", description)
        # print("Frame",Frame)
        # for text in dialog['story']:
        tokens = description.strip().split()
        sentence=[]
        story = []
        sentence.extend([self.story_vocab(token) for token in tokens])
        if len(sentence) > 23:
            sentence = sentence[:24]
        story.append(sentence)


        frame = []
        tmp_frame = []
        tmp_frame.extend([self.frame_vocab(frame) for frame in Frame ])
        if len(tmp_frame) > 23:
            tmp_frame = tmp_frame[:24]
        frame.append(tmp_frame)

        return sentence, tmp_frame

    def __len__(self):
        return len(self.dialogs)

def ROC_collate_fn(data):

    texts, frame = zip(*data)



    lengths = [len(x)+1 for x in texts]
    #max_seq_len = max(lengths)
    max_seq_len = 25
    texts = [[1] + s[:max_seq_len] + [2] + [0 for _ in range(max_seq_len - len(s) - 2)] for s in texts]
    texts_pos = [[pos_i+1 if w_i != 0 else 0
         for pos_i, w_i in enumerate(inst)] for inst in texts]


    frame_lengths = [len(x)+1 for x in frame]
    #max_frame_seq_len = max(frame_lengths)
    max_frame_seq_len = 25
    frame = [[1] + s + [2] + [0 for _ in range(max_frame_seq_len - len(s) - 2)] for s in frame]
    frame_pos = [[pos_i+1 if w_i != 0 else 0
         for pos_i, w_i in enumerate(inst)] for inst in frame]

    # print("texts",texts)
    # print("frame",frame)
    # print("lengths",max_seq_len)
    # print("max_frame_seq_len",max_frame_seq_len)
    targets = torch.LongTensor(texts).view(-1, max_seq_len)
    lengths = torch.LongTensor(lengths).view(-1,1)
    targets_pos = torch.LongTensor(texts_pos).view(-1, max_seq_len)
    frame = torch.LongTensor(frame).view(-1,max_frame_seq_len)
    frame_lengths = torch.LongTensor(frame_lengths).view(-1,1)
    frame_pos = torch.LongTensor(frame_pos).view(-1, max_seq_len)


    #return targets, lengths, frame, frame_lengths
    return frame, frame_pos, targets, targets_pos



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
