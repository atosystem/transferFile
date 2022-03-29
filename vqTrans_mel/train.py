
import os
import json
import yaml
import pickle
import datetime
import numpy as np
from collections import OrderedDict
import sys

import torch
from model import TransformerXL
from dataset.CodeWordsDataset import CodeWordsDataset
from dataset.vocab import Vocab

myvocab = Vocab()


def main():
    config_path = None
    if len(sys.argv) == 2:
        config_path = sys.argv[1]
    # gen config
    modelConfig, trainConfig = get_configs(config_path=config_path)

    # load dictionary
    event2word, word2event = myvocab.id2token, myvocab.token2id

    # load train data
    # training_data = np.load(os.path.join(trainConfig['ROOT'],'train_data_XL.npz'))

    train_dataset = CodeWordsDataset(data_path=trainConfig["selected_songs_json"],
        midiDIR=trainConfig["midiDIR"],
        partition="train",
        max_length=trainConfig["seq_len"],
        grp_size = trainConfig["group_size"]
        )
    dev_dataset = CodeWordsDataset(data_path=trainConfig["selected_songs_json"],
        midiDIR=trainConfig["midiDIR"],
        partition="dev",
        max_length=trainConfig["seq_len"],
        grp_size = trainConfig["group_size"]
        )
    # val_dataset = CodeWordsDataset(data_path=trainConfig["val_data"])

    device = torch.device("cuda:{}".format(trainConfig['gpuID']) if not trainConfig["no_cuda"] and torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = trainConfig['gpuID']

    print('Device to train:', device)
    
    resume = trainConfig['resume_training_model']

    # declare model
    model = TransformerXL(
            modelConfig,
            device,
            event2word=event2word, 
            word2event=word2event, 
            is_training=True)

    # train
    model.train(train_dataset,
                trainConfig,
                device,
                resume,
                dev_data=dev_dataset)
            

def get_configs(config_path="config.yml"):
    cfg = yaml.full_load(open(config_path, 'r')) 

    modelConfig = cfg['MODEL']
    trainConfig = cfg['TRAIN']

    cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if trainConfig['exp_tag']=="":
        trainConfig['exp_tag'] = cur_date
    experiment_Dir = os.path.join(trainConfig['output_dir'],trainConfig['exp_tag'])
    if not os.path.exists(experiment_Dir):
        print('experiment_Dir:', experiment_Dir)
        os.makedirs(experiment_Dir)
    else:
        print("exp dir exists ({})!".format(experiment_Dir))
        ans = input("Delete? [y/n]")
        if ans.strip().lower() == "y":
            import shutil
            if os.path.exists(os.path.join(experiment_Dir,"tfb_log")):
                shutil.rmtree(os.path.join(experiment_Dir,"tfb_log"))
            shutil.rmtree(experiment_Dir)
            os.makedirs(experiment_Dir)
        else: exit(1)
    
    print('Experiment: ', experiment_Dir)
    trainConfig.update({'experiment_Dir': experiment_Dir})


    with open(os.path.join(experiment_Dir, 'config.yml'), 'w') as f:
        doc = yaml.dump(cfg, f)

    print('='*5, 'Model configs', '='*5)
    print(json.dumps(modelConfig, indent=1, sort_keys=True))
    print('='*2, 'Training configs', '='*5)
    print(json.dumps(trainConfig, indent=1, sort_keys=True))
    return modelConfig, trainConfig


if __name__ == '__main__':
    main()


