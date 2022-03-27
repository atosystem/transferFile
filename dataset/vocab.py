# using subbeat shift and motif_start, motif_end label
import pickle
from miditoolkit import midi
import numpy as np
import chorder
import miditoolkit
import os
import math
from chorder import dechorder
from miditoolkit.midi import parser as mid_parser  
from miditoolkit.midi import containers as ct
import copy
import random
import json
import torch

GM1_inst_groups_FP = "/home/atosystem/nas_disk/projects/multitrack/remi-xl_remiMulti/dataset/GM1_inst_groups.json"

with open(GM1_inst_groups_FP) as f:
    GM1_inst_groups = json.load(f)

# -1 to match with miditoolkit setting
for i in range(len(GM1_inst_groups["instruments_groups"])):
  GM1_inst_groups["instruments_groups"][i]["programs"] = [x-1 for x in GM1_inst_groups["instruments_groups"][i]["programs"]]

inst2GrpID = {}
for i,inst_grp in enumerate(GM1_inst_groups["instruments_groups"]):
    for n in inst_grp["programs"]:
        inst2GrpID[n] = i

inst2GrpID = { k:inst2GrpID[k]  for k in sorted(inst2GrpID.keys())}

grpID2name = { i:inst_grp["name"] for i,inst_grp in enumerate(GM1_inst_groups["instruments_groups"])}
grp2Inst = { g["name"] : g["programs"][0] for g in GM1_inst_groups["instruments_groups"]}

class Vocab(object):
    num2pitch = {
        0: 'C',
        1: 'C#',
        2: 'D',
        3: 'D#',
        4: 'E',
        5: 'F',
        6: 'F#',
        7: 'G',
        8: 'G#',
        9: 'A',
        10: 'A#',
        11: 'B',
    }
    pitch2num = { v:k for k,v in num2pitch.items()}
    def __init__(self) -> None:
        
        self.token2id = {}
        self.id2token = {}

        self.chroma_dim = 32
        self.track_dim = 16
        self.other_dim=32

        self.n_chroma = 128
        self.n_track = 128
        self.n_other = 128

        self.codeBook_Names = ["Chroma","Track","Other"]

        self.n_tokens = 0

        self.token_type_base = {}

        self.build()

        # vocab
        # Note-On (129) : 0 (padding) ,1 ~ 127(highest pitch) , 128 (rest)
        # Note-Duration : 1 ~ 16 beat * 3
        # min resulution 1/12 notes        

    def build(self):
        # self.word2id = {}
        # self.id2word = {}
        self.token2id = {}
        self.id2token = {}
        
        self.n_tokens = 0

        self.token2id['padding'] = 0
        self.n_tokens += 1

        # SOS
        self.token_type_base = {'SOS' : 1}
        self.token2id[ 'SOS' ] = self.n_tokens
        self.n_tokens += 1

        # EOS
        self.token_type_base = {'EOS' : self.n_tokens}
        self.token2id[ 'EOS' ] = self.n_tokens
        self.n_tokens += 1

        # chroma
        self.token_type_base['Chroma'] = self.n_tokens
        for _i in range(self.n_chroma):
            self.token2id[ 'Chroma_{}'.format(_i) ] = self.n_tokens
            self.n_tokens += 1

        # track
        self.token_type_base['Track'] = self.n_tokens
        for _i in range(self.n_track):
            self.token2id[ 'Track_{}'.format(_i) ] = self.n_tokens
            self.n_tokens += 1

        # other
        self.token_type_base['Other'] = self.n_tokens
        for _i in range(self.n_other):
            self.token2id[ 'Other_{}'.format(_i) ] = self.n_tokens
            self.n_tokens += 1
        
        self.n_tokens = len(self.token2id)

        for w , v in self.token2id.items():
            self.id2token[v] = w

    def loadPkl(self,pkl_path,verbose=False):
        with open(pkl_path,'rb') as f:
            data = pickle.load(f)
        bar_enc_ids: torch.Tensor = data["bar_enc_ids"]
        bar_enc_ids = bar_enc_ids.flatten()
        tokens = [ "{}_{}".format(self.codeBook_Names[i%3],x) for i,x in enumerate(bar_enc_ids)]
        tokens = ["SOS"] + tokens + ["EOS"]
        tokenIDs = [self.token2id[x] for x in tokens]

        return tokenIDs
        
       
    def toPkl(self,tokenIDs,output_pkl_path,verbose=False):
        tokens = [self.id2token[x] for x in tokenIDs]
        assert tokens[0] == "SOS" and tokens[-1] == "EOS", "must start with SOS and end with EOS {}".format((tokens[0],tokens[-1]))

        tokens = tokens[1:-1]
        assert len(tokens) % 3 ==0, "len of tokens must be multiple of 3"

        for i in range(len(tokens)):
            assert tokens[i].startswith(self.codeBook_Names[i%3]), "Error pattern at #{} {}, must be {}".format(i,tokens[i],self.codeBook_Names[i%3])

        tokens = [ int(x.split("_")[-1]) for x in tokens]

        bar_enc_ids = torch.Tensor(tokens)
        bar_enc_ids = bar_enc_ids.reshape(-1,3)

        with open(output_pkl_path,"wb") as f:
            pickle.dump({"bar_enc_ids":bar_enc_ids},f)

   
    def dumpToText(self,seqIDs: list) -> str:
        _s: list = [ self.id2token[x] for x in seqIDs]
        return " ".join(_s)

    def __str__(self):
        ret = ""
        for w,i in self.token2id.items():
            ret = ret + "{} : {}\n".format(w,i)

        for i,w in self.id2token.items():
            ret = ret + "{} : {}\n".format(i,w)

        ret += "\nTotal events #{}".format(len(self.id2token))

        return ret

    def __repr__(self):
        return self.__str__() 

    def __len__(self):
        return len(self.token2id)

if __name__ == '__main__':
    myvocab = Vocab()
    # print(myvocab)
    print(len(myvocab))
    t = myvocab.loadPkl("../../../YJ_vq_trans/vqBarEncs/dev/lmd_16364.pkl")
    print(t[:10])
    myvocab.toPkl(t,"asd.pkl")
    exit(1)
    t = myvocab.parseMidi("../../mydataset/LMDfull_dataset_midi_with_chords_selected/lmd_1_full.mid",False)
    myvocab.toMidi("asd.mid",tokenIDs=t)
    exit()
    with open("asd.pkl",'rb') as f:
        xx = pickle.load(f)
    xx_s = myvocab.dumpToText(xx)
    print(xx_s)
    myvocab.toMidi("asd.mid",tracks=None,eventIds=xx)
    exit()
    # myvocab.build()
    
    # print(myvocab)
    t = myvocab.parseMidi("../mydataset/LMDfull_dataset_midi_with_chords_selected/lmd_2_full.mid")
    myvocab.toMidi("asd.mid",tracks=t)
    exit()
    ret = myvocab.midi2REMI("/home/atosystem/nas_disk/projects/clustering/contrastive_pop/exp_outputs/001.mid",trim_intro=True,verbose=True)
    # _ , ret = data_pitch_augment(myvocab,[],ret)
    myvocab.REMIID2midi(ret,"aaa.mid")
    ret = myvocab.preprocessREMI(ret,max_seq_len=512)
    s = 0
    for x in ret["tgt_segments"]:
        if myvocab.token2id["Theme_Start"] in x:
            s +=1
        
    print(s,len(ret["tgt_segments"]))

    # print(len(ret["src"]))
    
    # print(ret)
    # myvocab.vocabID2midi(event_ids=ret,midi_path="asd.mid",verbose=True)
    # print(myvocab.decompose_shift(50))
    # /home/atosystem/nas_disk/dataset/hundrum_original/Beethoven/Op022-01.mid


