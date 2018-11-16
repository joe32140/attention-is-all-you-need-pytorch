import ROC_loader as ROC_loader
import pickle
from build_vocab import Vocabulary
import torch
class Loaders():

    def __init__(self):
        self.loader ={}
        with open("../data/ROC_Frame_vocab.pkl",'rb') as f:
            self.frame_vocab = pickle.load(f)
        with open("../data/ROC_Story_vocab.pkl",'rb') as f:
            self.story_vocab = pickle.load(f)


    def get_loaders(self, args):

        STORY_FRAME_PATH = "/home/cloud60138/data/ROC_Story_fivesentwith_onlyVerbframe_lu_{}_filtered.json"


        self.loader['train'] = ROC_loader.get_ROC_loader(STORY_FRAME_PATH.format('train'),
                                                         self.story_vocab,
                                                         self.frame_vocab,
                                                         args.batch_size,
                                                         True, 5)
        self.loader['val'] = ROC_loader.get_ROC_loader(STORY_FRAME_PATH.format('valid'),
                                                         self.story_vocab,
                                                         self.frame_vocab,
                                                         args.batch_size,
                                                         True, 5)
        self.loader['test'] = ROC_loader.get_ROC_loader(STORY_FRAME_PATH.format('test'),
                                                         self.story_vocab,
                                                         self.frame_vocab,
                                                         args.batch_size,
                                                         False, 5)

        STORY_FRAME_PATH = "../data/COCO_desription_with_existed_verb_frame_mappedSeq.json"

        Dataset = ROC_loader.ROCDataset(self.story_vocab,
                         self.frame_vocab,
                         text_path=STORY_FRAME_PATH)

        self.loader['coco'] = ROC_loader.get_COCO_loader(Dataset,
                                                         self.story_vocab,
                                                         self.frame_vocab,
                                                         args.batch_size//8,
                                                         True, 5)
