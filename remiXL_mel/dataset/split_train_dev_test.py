import json

from importlib_metadata import os

SELECTED_SONGS_JSON_FP = "./selected_songs_id_new.json"
SONG_MEATA_DIR = "/home/atosystem/nas_disk/projects/multitrack/mydataset/metaForMidi"


with open(SELECTED_SONGS_JSON_FP,"r") as f:
    selected_songs_dict = json.load(f)


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



# print(selected_songs_dict)

print("Total #{} midis".format(len(selected_songs_dict["all_selected_songs"])))
total = len(selected_songs_dict["all_selected_songs"])

TRAIN_RATIO = 0.9
DEV_RATIO = 0.05
TEST_RATIO = 0.05

mel_inst_hist = { k:[] for k in range(len(grpID2name))}

selected_songs_dict["all_selected_songs"] = sorted(selected_songs_dict["all_selected_songs"],key= lambda x: int(x.replace("lmd_","")))

for s in selected_songs_dict["all_selected_songs"]:
    print(s)
    meta_fp = os.path.join(SONG_MEATA_DIR,"{}.json".format(s))
    with open(meta_fp,"r") as f:
        _meta = json.load(f)
    mel_inst_hist[inst2GrpID[_meta["melody_instrument"]]].append(s)

# print([ len(mel_inst_hist[k]) / len(selected_songs_dict["all_selected_songs"])  for k in mel_inst_hist])

# all: [0.09025733906874953, 0.012602822428495962, 0.06429703418609917, 0.07478680854275149, 0.1322919024979247, 0.050864085729378915, 0.06671194626820617, 0.07335295449400045, 0.16028978944985284, 0.1713832918270319, 0.08172968077880914, 0.02143234472869972, 0.0, 0.0, 0.0, 0.0]
# train: [0.09027661357921207, 0.012657166806370494, 0.06429170159262364, 0.07476948868398994, 0.13227158424140822, 0.05088013411567477, 0.06672254819782061, 0.07334450963956413, 0.16026823134953896, 0.17133277451802179, 0.08172673931265717, 0.02145850796311819, 0.0, 0.0, 0.0, 0.0]
# val: [0.09036144578313253, 0.012048192771084338, 0.06475903614457831, 0.07530120481927711, 0.13253012048192772, 0.05120481927710843, 0.06626506024096386, 0.07379518072289157, 0.15963855421686746, 0.1716867469879518, 0.08132530120481928, 0.02108433734939759, 0.0, 0.0, 0.0, 0.0]
# test: [0.0898021308980213, 0.0121765601217656, 0.0639269406392694, 0.0745814307458143, 0.1324200913242009, 0.0502283105022831, 0.0669710806697108, 0.0730593607305936, 0.1613394216133942, 0.1719939117199391, 0.0821917808219178, 0.0213089802130898, 0.0, 0.0, 0.0, 0.0]


selected_songs_dict["train"] = []
selected_songs_dict["dev"] = []
selected_songs_dict["test"] = []

for k in mel_inst_hist:
    total = len(mel_inst_hist[k])
    for id,song in enumerate(mel_inst_hist[k]):
        if id <= TRAIN_RATIO * total:
            selected_songs_dict["train"].append(song)
        elif id <= (TRAIN_RATIO+DEV_RATIO) * total:
            selected_songs_dict["dev"].append(song)
        else:
            selected_songs_dict["test"].append(song)


total = len(selected_songs_dict["all_selected_songs"])
print("train\t#{} {:.3f}%".format(len(selected_songs_dict["train"]),len(selected_songs_dict["train"])/total*100))
print("dev\t#{} {:.3f}%".format(len(selected_songs_dict["dev"]),len(selected_songs_dict["dev"])/total*100))
print("test\t#{} {:.3f}%".format(len(selected_songs_dict["test"]),len(selected_songs_dict["test"])/total*100))


with open(SELECTED_SONGS_JSON_FP,"w") as f:
    json.dump(selected_songs_dict,f)

# train   #11930 90.031%
# dev     #664 5.011%
# test    #657 4.958%