import numpy as np
import pickle
import os
import glob
import miditoolkit
import tqdm
import vocab
from multiprocessing import Pool
from multiprocessing import Manager

midi_fps = sorted(glob.glob("../../mydataset/LMDfull_dataset_midi_with_chords_selected/*_full.mid"))
output_dir = "../../mydataset/midiFullPkls"

myvocab = vocab.Vocab()

manager = Manager()
seq_len = manager.list([])

# features = manager.dict(features)


def process_one(fp):
    # if os.path.exists(os.path.join(output_dir,os.path.basename(fp).replace(".mid",".pkl"))):
    #     return
    # print(os.path.join(output_dir,os.path.basename(fp).replace(".mid",".pkl")))
    tokenIDs = myvocab.parseMidi(fp)
    with open(os.path.join(output_dir,os.path.basename(fp).replace(".mid",".pkl")),"wb") as f:
        pickle.dump(tokenIDs,f,pickle.HIGHEST_PROTOCOL)

    seq_len.append( len(tokenIDs))

    print("{} done".format(os.path.basename(fp)))

# for fp in tqdm.tqdm(midi_fps):

if not os.path.exists("stat_pkl.npz"):
    print("Collect data")
    # collect data from all mids
    pool = Pool(processes=48)
    args = []

    # for fp in tqdm.tqdm(midi_fps):
    for fp in midi_fps:
        # elapsed += 1
        # print ('>> now processing: {} ({} of {})'.format(m, elapsed, len(midis)))
        args.append((fp,))

    pool.starmap(process_one, args)
    print("Done")
    # print(features)
    seq_len = np.array(list(seq_len))


    features = {
            "seq_len": seq_len
        }

    np.savez("stat_pkl.npz",**features)
    print("Done saving file")

else:
    
    features = np.load("./stat_pkl.npz")
    from matplotlib import pyplot as plt
    from scipy.stats import mode
    f = "seq_len"
    print("Draw {}".format(f))
    # plt.hist(features[f],bins=np.arange(start=0,stop = 5,step=0.25))
    plt.hist(features[f],bins=50)
    plt.title(f)
    plt.savefig("{}/{}.png".format(".",f))
    plt.clf()
    print("max:{}".format(np.max(features[f])))
    print("min:{}".format(np.min(features[f])))
    print("avg:{}".format(np.average(features[f])))
    print("std:{}".format(np.std(features[f])))
    print("mode:{}".format(mode(features[f])))
    for i in range(41):
        print("percent < 1024*{}: {}".format(i,len([x for x in features[f] if x <= 1024*i]) / len(features[f])))
    print("=" * os.get_terminal_size().columns)


