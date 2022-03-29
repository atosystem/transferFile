import json
from random import randint, random
from typing import Tuple
import torch
import glob
import pickle
import os
from .vocab import Vocab


myvocab = Vocab()

class CodeWordsDataset(torch.utils.data.dataset.Dataset):
    def __init__(self,data_path,midiDIR="../../../YJ_vq_trans/vqBarEncs",partition="train",max_length=1024,grp_size = 13):
        self.data_path: str = data_path
        if  not self.data_path.endswith(".json"):
            print("Please input .json file")
            exit(1)
        with open(self.data_path) as f:
            songs_dict = json.load(f)
        

        self.pkl_files = sorted(glob.glob(os.path.join(midiDIR,partition,"*.pkl")))
        print("{}:#{}".format(partition,len(self.pkl_files)))

        # sanity check for files
        # for x in self.pkl_files:
        #     if not os.path.exists(x):
        #         print("Error not exist: {}".format(x))
        #         exit(1)
        self.max_length: int = max_length
        self.grp_size = grp_size
        self.padding_idx = 0

    def __getitem__(self, idx):
        fp = self.pkl_files[idx]
        toks = myvocab.loadPkl(fp)

        att_msk, _seq = self.padding(toks,total_length=self.grp_size*self.max_length+1)

        return {
            "att_msk" : torch.tensor(att_msk),
            "src": torch.tensor(_seq)
        }

    def random_offset(self,seq,total_length):
        if len(seq) < total_length:
            return seq
        else:
            st_idx = randint(0,len(seq)-total_length)
            seq = seq[st_idx:st_idx+total_length]
            assert len(seq) == total_length
            return seq

    def padding(self,seq: list, total_length: int = None) -> Tuple:
        if total_length == None:
            total_length = self.max_length
        seq = self.random_offset(seq,total_length=total_length)
        att_msk = [1] * min(total_length,len(seq)) + [0] * max(0,total_length -  len(seq))
        _seq = seq[:total_length] + [self.padding_idx] *  max(0,total_length - len(seq) )
        # print(len(att_msk))
        assert len(att_msk) == total_length
        assert len(_seq) == total_length
        return att_msk,_seq

    def __len__(self):
        return len(self.pkl_files)
    
