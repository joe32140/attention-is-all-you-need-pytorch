import os
import nltk
import pickle
import json
import argparse
from collections import Counter
import numpy as np
from transformer import Constants

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.word_count = []

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_glove_voc(threshold, vocab, paragraph):
    data_path = '/corpus/glove/pretrained_vector/english/glove.42B.300d.{}'
    with open(data_path.format('json'),'r', encoding='utf-8') as f:
        glove = json.load(f, encoding='utf-8')

    count = 0
    word2vec = {}
    if paragraph:
        weight_matrix = np.random.uniform(-0.5, 0.5, size=(threshold+6,300))
    else:
        weight_matrix = np.random.uniform(-0.5, 0.5, size=(threshold+4,300))

    with open(data_path.format('txt'),'r', encoding='utf8') as f:
        for line in f:
            l = line.strip().split()
            word = l[0]
            if vocab(word) != 3:
                weight_matrix[vocab(word),:] = np.asarray(list(map(float, l[1:])))

            count += 1


    return weight_matrix

def build_vocab(text, threshold, paragraph):
    """Build a simple vocabulary wrapper."""
    dialog = json.load(open(text[0], 'r'))
    print("Check")
    counter = Counter()
    for i, entry in enumerate(dialog):
        candidate_frame = ""
        #print("Frame",dialog[entry]['Frame'])
        #print("LU",dialog[entry]['LU'])
        if args.parse:
            description = entry[0]['parse_template'] + ' ' + entry[0]['text']
        else:
            frames = dialog[entry]['Frame']
            lus =  dialog[entry]['LU']
        for j, lu in enumerate(lus):
            if lu[-2:] ==".v":
                candidate_frame = frames[j]
                break
        #print("candidate_frame", candidate_frame)

        tokens = [candidate_frame] #subject? verb? object? modifier?

        #exit()
        counter.update(tokens)

        #if i % 1000 == 0:
         #   print("[%d/%d] Tokenized the captions." %(i, len(dialog['annotations'])))

    #print("counter", counter.items())
    # If the word frequency is less than 'threshold', then the word is discarded.
    """ Story Frame loading """
    dialog1 = json.load(open("/home/cloud60138/event/frame_data/frame_clean_train.json",'r'))
    dialog2 = json.load(open("/home/cloud60138/event/frame_data/frame_clean_val.json",'r'))
    dialog3 = json.load(open("/home/cloud60138/event/frame_data/frame_clean_test.json",'r'))
    dialogs = [dialog1, dialog2, dialog3]
    for dialog in dialogs:
        for i, entry in enumerate(dialog):
            tmp_frame = []
            if args.parse:
                description = entry[0]['parse_template'] + ' ' + entry[0]['text']
            else:
                lus = entry[0]['LU']
                frames = entry[0]['Frame']
            for j , lu in enumerate(lus):
                if lu[-2:] == ".v":
                    tmp_frame.append(frames[j])
                    print("Framesss", frames[j])
            tokens = tmp_frame  #subject? verb? object? modifier?
            counter.update(tokens)






    words = [word for word, cnt in counter.items() if cnt >= threshold]
    # Creates a vocab wrapper and add some special .
    vocab = Vocabulary()
    word_count = {}
    for word, cnt in counter.items():
        word_count[word] = cnt
    print("word_count",word_count)
    vocab.add_word(Constants.PAD_WORD)
    vocab.add_word(Constants.BOS_WORD)
    vocab.add_word(Constants.BOS_WORD2)
    vocab.add_word(Constants.BOS_WORD3)
    vocab.add_word(Constants.BOS_WORD4)
    vocab.add_word(Constants.BOS_WORD5)
    vocab.add_word(Constants.EOS_WORD)
    vocab.add_word(Constants.UNK_WORD)

    if paragraph:
        vocab.add_word('<start1>')
        vocab.add_word('<start2>')
    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    for word, idx in vocab.word2idx.items():
        if word =='<pad>' or word =='<start>' or word=='<end>' or word=='<unk>':
            vocab.word_count.append(int(1))
        else:
            count = word_count[word]
            vocab.word_count.append(1/count)
    return vocab

def main(args):
    if not os.path.exists(args.vocab_dir):
        os.makedirs(args.vocab_dir)
        print("Make Data Directory")
    vocab = build_vocab(text=[args.caption_path],
                        threshold=args.threshold, paragraph=args.paragraph)
    #W = build_glove_voc(len(vocab), vocab, args.paragraph)
    vocab_path = os.path.join(args.vocab_dir, 'description_story_frame_vocab.pkl')
    #weight_path = os.path.join(args.vocab_dir, 'W.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    #with open(weight_path, 'wb') as f:
    #    pickle.dump(W, f)

    print("Total vocabulary size: %d" %len(vocab))
    print(vocab.word2idx)
    #print(vocab.word_count)
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str,
                        default='/home/cloud60138/data/description_frameAndlu.json',
                        help='path for train annotation file')
    #parser.add_argument('--caption_path2', type=str,
    #                    default='../event_data/event_clean_val.json',
    #                    help='path for train annotation file')
    #parser.add_argument('--caption_path3', type=str,
    #                    default='../event_data/event_clean_test.json')
    parser.add_argument('--vocab_dir', type=str, default='../../data/',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4,
                        help='minimum word count threshold')
    parser.add_argument('--paragraph', action='store_true', default=False,
                        help='minimum word count threshold')
    parser.add_argument('--parse', action='store_true', default=False,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
