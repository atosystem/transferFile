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

GM1_inst_groups_FP = "./dataset/GM1_inst_groups.json"

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
# assign melody with Pipe
grp2Inst.update({"Melody":72})


def compare_intersect(x, y):
    return frozenset(x).intersection(y)

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

        # split each beat into self.q_bar subbeats
        self.q_beat: int = 4

        # midi pitch 0 (padding) ,1 ~ 127(highest pitch) 
        self._pitch_bins: np.ndarray = np.arange(start=1,stop=128)
        
        # midi program number
        # self._inst_bins: np.ndarray = np.arange(start=0,stop=128)

        # about 98% of the notes in LMD <= 1 bar length
        self._duration_bins = np.arange(start=1,stop=self.q_beat*4+1)

        self._velocity_bins =  np.arange(start=1,stop=127+1)

        self._tempo_bins = np.arange(start=30,stop=197,step=3)

        self._position_bins = np.arange(start=0,stop=self.q_beat*4)

        # track group
        self.track_names = ["Drum","Piano","Ensemble","Bass","Guitar","Brass","Pipe","Reed","Synth_Lead"]

        # chords
        self.chord_roots = list(Vocab.pitch2num.keys())
        self.chord_qualities = chorder.Chord.qualities


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

        
        # Note-On
        self.token_type_base = {'Note-On' : 1}
        for i in self._pitch_bins:
            self.token2id[ 'Note-On_{}'.format(i) ] = self.n_tokens
            self.n_tokens += 1
        
        

        # Note-Duration
        self.token_type_base['Note-Duration'] = self.n_tokens
        for note_dur in self._duration_bins:
            self.token2id[ 'Note-Duration_{}'.format(note_dur) ] = self.n_tokens
            self.n_tokens += 1


        # Note-Velocity
        self.token_type_base['Note-Velocity'] = self.n_tokens
        for vel in self._velocity_bins:
            self.token2id[ 'Note-Velocity_{}'.format(vel) ] = self.n_tokens
            self.n_tokens += 1
        
        # Tempo
        self.token_type_base['Tempo'] = self.n_tokens
        for tmp in self._tempo_bins:
            self.token2id[ 'Tempo_{}'.format(tmp) ] = self.n_tokens
            self.n_tokens += 1

        # Positions
        self.token_type_base['Position'] = self.n_tokens
        for pos in self._position_bins:
            self.token2id[ 'Position_{}'.format(pos) ] = self.n_tokens
            self.n_tokens += 1

        # bar start and end
        self.token_type_base['Bar'] = self.n_tokens
        self.token2id[ 'Bar-Start' ] = self.n_tokens
        self.n_tokens += 1
        # self.token_type_base['Bar'] = self.n_tokens
        # self.token2id[ 'Bar-End' ] = self.n_tokens
        # self.n_tokens += 1

        # # track start and end
        # self.token_type_base['Track'] = self.n_tokens
        # self.token2id[ 'Track-Start' ] = self.n_tokens
        # self.n_tokens += 1
        # self.token_type_base['Track'] = self.n_tokens
        # self.token2id[ 'Track-End' ] = self.n_tokens
        # self.n_tokens += 1

        # piece start and end
        self.token_type_base['Piece'] = self.n_tokens
        self.token2id[ 'Piece-Start' ] = self.n_tokens
        self.n_tokens += 1
        self.token_type_base['Piece'] = self.n_tokens
        self.token2id[ 'Piece-End' ] = self.n_tokens
        self.n_tokens += 1
        
        # track names
        self.token_type_base['Track'] = self.n_tokens
        for t in self.track_names:
            self.token2id[ 'Track_{}'.format(t) ] = self.n_tokens
            self.n_tokens += 1
        
        self.n_tokens = len(self.token2id)

        # chord events
        self.token_type_base['Chord'] = self.n_tokens
        for t in self.chord_roots:
            self.token2id[ 'Chord-Root_{}'.format(t) ] = self.n_tokens
            self.n_tokens += 1
        
        for t in self.chord_qualities:
            self.token2id[ 'Chord-Quality_{}'.format(t) ] = self.n_tokens
            self.n_tokens += 1
        
        self.n_tokens = len(self.token2id)

        for w , v in self.token2id.items():
            self.id2token[v] = w

    def getPitch(self,input_event):
        """Return corresponding note pitch
        if input_event is not a note, it returns -1

        Args:
            input_event (str or int): REMI Event Name or vocab ID
        """
        if isinstance(input_event,int):
            input_event = self.id2token[input_event]
        elif isinstance(input_event,str):
            pass
        else:
            try:
                input_event = int(input_event)
                input_event = self.id2token[input_event]
            except:
                raise TypeError("input_event should be int or str, input_event={}, type={}".format(input_event,type(input_event)))
        
        if not input_event.startswith("Note-On"):
            return -1

        # prefix = input_event.split("_")[0]
        # base = self.token_type_base[prefix]
        # pitch = int(input_event.split("_")[1])
        # return self._pitch_bins[pitch - base]
        assert int(input_event.split("_")[1]) >=1 and int(input_event.split("_")[1]) <=127
        return int(input_event.split("_")[1])
        
    def parseMidi(self,midi_path,verbose=False):
        midi_obj: miditoolkit.midi.MidiFile = mid_parser.MidiFile(midi_path)
        output_tracks = []
        assert midi_obj.time_signature_changes[0].denominator == 4
        assert midi_obj.time_signature_changes[0].numerator == 4
        bar_edges = np.arange(start=0,stop = midi_obj.max_tick,step=midi_obj.ticks_per_beat*4)
        ticks_per_bar = midi_obj.ticks_per_beat * 4
        ticks_per_step = midi_obj.ticks_per_beat / self.q_beat
        all_events_grps =  []
        for i in range(len(bar_edges)-1): all_events_grps.append([])
        tempo_id = (np.abs(self._tempo_bins - midi_obj.tempo_changes[0].tempo)).argmin()
        # collect tracks' note
        for midi_track in midi_obj.instruments:
            assert midi_track.name in self.track_names

            _track_name = midi_track.name
            for bar_index, bar_start_tick in enumerate(bar_edges[:-1]):
                if verbose:
                    print("Track:{} Bar #{}, start at tick:{}".format(_track_name,bar_index,bar_start_tick))

                notes = list(filter(lambda n: n.start >= bar_start_tick and n.start < bar_start_tick + ticks_per_bar and n.pitch > 0,midi_track.notes))
                notes = sorted(notes,key= lambda n: (n.start,-n.pitch))
                for n in notes:
                    if verbose:
                        print(n)
                    pos = int((n.start - bar_start_tick)  // ticks_per_step)
                    assert not (pos == 16)
                    all_events_grps[bar_index].append({
                        "type" : "note",
                        "pos" : pos,
                        "pitch" : n.pitch,
                        "dur" : min(int((n.end-n.start)//ticks_per_step),len(self._duration_bins)-1),
                        "vel" : n.velocity,
                        "track_name" : _track_name,
                        "priority" : 1 + self.track_names.index(_track_name)
                    })
        midi_obj.markers = sorted(midi_obj.markers,key=lambda x: x.time)
        # # print(midi_obj.lyrics)
        # print(midi_obj.markers[-1])
        # print(bar_edges[-2:])
        # print(len(midi_obj.markers))
        # print(len(bar_edges))
        # exit()
        # collect chords
        for bar_index, bar_start_tick in enumerate(bar_edges[:-1]):
            if verbose:
                print("Chord Bar #{}, start at tick:{}".format(bar_index,bar_start_tick))
            chords = list(filter(lambda n: n.time >= bar_start_tick and n.time < bar_start_tick + ticks_per_bar,midi_obj.markers))
            for _chord_i,_chord in enumerate(chords):
                pos = int((_chord.time - bar_start_tick) // ticks_per_step)
                if not _chord.text.split('_')[0] == 'N':
                    all_events_grps[bar_index].append({
                        "type" : "chord",
                        "pos" : pos,
                        "root" : _chord.text.split('_')[0],
                        "quality" : _chord.text.split('_')[1],
                        "priority" : 0,
                        "pitch" :0
                    })
            # notes = sorted(notes,key= lambda n: (n.start,-n.pitch))

        all_words = []
        all_words.append("Piece-Start")
        all_words.append("Tempo_{}".format(self._tempo_bins[tempo_id]))
        # convert to events
        # for bar_index, bar_start_tick in enumerate(bar_edges[:-1]):
        for bar_index, grp in enumerate(all_events_grps):
            if verbose:
                print("Bar #{}".format(bar_index))
            last_pos = -1
            last_track_name = ""
            all_words.append("Bar-Start")
            grp = sorted(grp,key= lambda n: (n["pos"],n["priority"],n["pitch"]))
            for n in grp:
                if verbose:
                    print(n)
                pos = n["pos"]
                if not pos == last_pos:
                    all_words.append("Position_{}".format(pos))
                    last_pos = pos
                    last_track_name = ""
                if n["type"] == "note":
                    if not last_track_name == n["track_name"]:
                        last_track_name = n["track_name"]
                        all_words.append("Track_{}".format(n["track_name"]))
                    
                    all_words.append("Note-On_{}".format(n["pitch"]))
                    all_words.append("Note-Duration_{}".format(n["dur"]))
                    all_words.append("Note-Velocity_{}".format(n["vel"]))
                elif n["type"] == "chord":
                    all_words.append("Chord-Root_{}".format(n["root"]))
                    all_words.append("Chord-Quality_{}".format(n["quality"]))
                else:
                    print("Error item: {}".format(n))
                    exit(1)

        all_words.append("Piece-End")

        all_ids = [ self.token2id[w] for w in all_words]

        return all_ids

    def toMidi(self,midi_path,tokenIDs):
        newMidiObj = miditoolkit.midi.MidiFile()
        ticks_per_bar = newMidiObj.ticks_per_beat * 4
        ticks_per_step = newMidiObj.ticks_per_beat / self.q_beat
        
        all_instruments = {
            _inst_name : ct.Instrument(
                program=grp2Inst[_inst_name] if not _inst_name=="Drum" else 0,
                is_drum=(_inst_name=="Drum"),
                name=_inst_name
            ) for _inst_name in self.track_names
        }

        all_tokens = [ self.id2token[w] for w in tokenIDs]
        assert all_tokens[0] == "Piece-Start"
        assert all_tokens[1].startswith("Tempo")
        tempo = int(all_tokens[1].split("_")[1])
        newMidiObj.tempo_changes.append(ct.TempoChange(tempo = tempo,time = 0))
        all_tokens = all_tokens[2:]

        i = 0
        current_bar_start = - ticks_per_bar
        current_pos = 0
        last_track_name = ""
        while(i < len(all_tokens)):
            tok = all_tokens[i]
            if tok.startswith("Bar"):
                if tok == "Bar-Start":
                    current_bar_start += ticks_per_bar
                i += 1
            elif tok.startswith("Position"):
                current_pos = int(tok.split("_")[-1])
                last_track_name = ""
                i += 1
            elif tok.startswith("Track"):
                last_track_name = tok[6:]
                i += 1
            elif tok.startswith("Note-On"):
                # print(i,tok)
                if not all_tokens[i+1].startswith("Note-Duration"):
                    i += 1
                    print("Skip {} becasue {}".format(tok,all_tokens[i+1]))
                    continue
                if not all_tokens[i+2].startswith("Note-Velocity"):
                    i += 2
                    print("Skip {},{} becasue {}".format(tok,all_tokens[i+1],all_tokens[i+2]))
                    continue
                assert all_tokens[i+1].startswith("Note-Duration")
                assert all_tokens[i+2].startswith("Note-Velocity")
                if last_track_name == "":
                    print("No track event specified, skipped {} {} {}".format(tok,all_tokens[i+1],all_tokens[i+2]))
                    i += 3
                    continue

                pitch = int(tok.split("_")[-1])
                dur = int(all_tokens[i+1].split("_")[-1]) * ticks_per_step
                vec = int(all_tokens[i+2].split("_")[-1])
                all_instruments[last_track_name].notes.append(miditoolkit.midi.containers.Note(
                    velocity=vec,
                    pitch=pitch,
                    start=int(current_pos*ticks_per_step+current_bar_start),
                    end=int(current_pos*ticks_per_step+current_bar_start+dur)
                ))
                i += 3
            elif tok.startswith("Chord-Root"):
                # assert all_tokens[i+1].startswith("Chord-Quality")
                if not all_tokens[i+1].startswith("Chord-Quality"):
                    print("No Chord-Quality event specified, skipped {}".format(tok))
                    i += 1
                    continue
                newMidiObj.markers.append(
                    ct.Marker(text="{}_{}".format(tok.replace("Chord-Root_",""),tok.replace("Chord-Quality_","")),
                        time=int(current_pos*ticks_per_step+current_bar_start)
                        )
                )
                i+=2
            elif tok.startswith("Piece-End"):
                break
            else:
                print("Error tok:{}, skipped".format(tok))
                i+=1
            
        newMidiObj.instruments = [ all_instruments[inst_name] for inst_name in all_instruments]

        newMidiObj.dump(midi_path)

    def sampleFromTracks(self,tracks):
        _track_ids = [x for x in range(len(tracks))]
        random.shuffle(_track_ids)
        _out = []
        _out.append(self.token2id["Piece-Start"])
        for _tId in _track_ids:
            _out.append(self.token2id["Track-Start"])
            _out.extend(tracks[_tId])
            _out.append(self.token2id["Track-End"])
        _out.append(self.token2id["Piece-End"])

        return _out

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

class Vocab_wMelody_DenseTrack(object):
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

        # split each beat into self.q_bar subbeats
        self.q_beat: int = 4

        # midi pitch 0 (padding) ,1 ~ 127(highest pitch) 
        self._pitch_bins: np.ndarray = np.arange(start=1,stop=128)
        
        # midi program number
        # self._inst_bins: np.ndarray = np.arange(start=0,stop=128)

        # about 98% of the notes in LMD <= 1 bar length
        self._duration_bins = np.arange(start=1,stop=self.q_beat*4+1)

        self._velocity_bins =  np.arange(start=1,stop=127+1)

        self._tempo_bins = np.arange(start=30,stop=197,step=3)

        self._position_bins = np.arange(start=0,stop=self.q_beat*4)

        # track group
        self.track_names = ["Drum","Melody","Piano","Ensemble","Bass","Guitar","Brass","Pipe","Reed","Synth_Lead"]

        # chords
        self.chord_roots = list(Vocab.pitch2num.keys())
        self.chord_qualities = chorder.Chord.qualities


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

        
        # Note-On
        self.token_type_base = {'Note-On' : 1}
        for _track_name in self.track_names:
            for i in self._pitch_bins:
                self.token2id[ 'Note-On-{}_{}'.format(_track_name,i) ] = self.n_tokens
                self.n_tokens += 1
        
        

        # Note-Duration
        self.token_type_base['Note-Duration'] = self.n_tokens
        for note_dur in self._duration_bins:
            self.token2id[ 'Note-Duration_{}'.format(note_dur) ] = self.n_tokens
            self.n_tokens += 1


        # Note-Velocity
        self.token_type_base['Note-Velocity'] = self.n_tokens
        for vel in self._velocity_bins:
            self.token2id[ 'Note-Velocity_{}'.format(vel) ] = self.n_tokens
            self.n_tokens += 1
        
        # Tempo
        self.token_type_base['Tempo'] = self.n_tokens
        for tmp in self._tempo_bins:
            self.token2id[ 'Tempo_{}'.format(tmp) ] = self.n_tokens
            self.n_tokens += 1

        # Positions
        self.token_type_base['Position'] = self.n_tokens
        for pos in self._position_bins:
            self.token2id[ 'Position_{}'.format(pos) ] = self.n_tokens
            self.n_tokens += 1

        # bar start and end
        self.token_type_base['Bar'] = self.n_tokens
        self.token2id[ 'Bar-Start' ] = self.n_tokens
        self.n_tokens += 1
        # self.token_type_base['Bar'] = self.n_tokens
        # self.token2id[ 'Bar-End' ] = self.n_tokens
        # self.n_tokens += 1

        # # track start and end
        # self.token_type_base['Track'] = self.n_tokens
        # self.token2id[ 'Track-Start' ] = self.n_tokens
        # self.n_tokens += 1
        # self.token_type_base['Track'] = self.n_tokens
        # self.token2id[ 'Track-End' ] = self.n_tokens
        # self.n_tokens += 1

        # piece start and end
        self.token_type_base['Piece'] = self.n_tokens
        self.token2id[ 'Piece-Start' ] = self.n_tokens
        self.n_tokens += 1
        self.token_type_base['Piece'] = self.n_tokens
        self.token2id[ 'Piece-End' ] = self.n_tokens
        self.n_tokens += 1
        
        # # track names
        # self.token_type_base['Track'] = self.n_tokens
        # for t in self.track_names:
        #     self.token2id[ 'Track_{}'.format(t) ] = self.n_tokens
        #     self.n_tokens += 1
        
        # self.n_tokens = len(self.token2id)

        # chord events
        self.token_type_base['Chord'] = self.n_tokens
        for t in self.chord_roots:
            self.token2id[ 'Chord-Root_{}'.format(t) ] = self.n_tokens
            self.n_tokens += 1
        
        for t in self.chord_qualities:
            self.token2id[ 'Chord-Quality_{}'.format(t) ] = self.n_tokens
            self.n_tokens += 1
        
        self.n_tokens = len(self.token2id)

        for w , v in self.token2id.items():
            self.id2token[v] = w
            
    def getMelodyTrack(self,midiobj:miditoolkit.MidiFile):
        assert len(midiobj.lyrics) > 0, "lyrics should exists"
        timestep = int(midiobj.ticks_per_beat // 4)
        melody_onset_times = [ int( round(n.time / timestep)*timestep )  for n in midiobj.lyrics]
        intersect_scores = []

        ins_names = [x.name for x in midiobj.instruments ]
        for ins in midiobj.instruments:
            if ins.is_drum is True: 
                intersect_scores.append(0)
                continue
            # _ins_onset_times = [ int( round(n.start / timestep)*timestep ) for n in ins.notes]
            _ins_onset_times = [ (n.start  ) for n in ins.notes]
            intersect_scores.append( len(compare_intersect(melody_onset_times,_ins_onset_times)) / len(melody_onset_times))
        # print(intersect_scores)
        # print(ins_names)
        best_match_id = np.argmax(intersect_scores)
        # print("best_match_id",best_match_id)

        if intersect_scores[best_match_id] < 0.6:
            print(intersect_scores)
            print(ins_names)
            return None, None
        assert intersect_scores[best_match_id] >= 0.6, intersect_scores
        return best_match_id, ins_names[best_match_id]
        

    def getPitch(self,input_event):
        """Return corresponding note pitch
        if input_event is not a note, it returns -1

        Args:
            input_event (str or int): REMI Event Name or vocab ID
        """
        if isinstance(input_event,int):
            input_event = self.id2token[input_event]
        elif isinstance(input_event,str):
            pass
        else:
            try:
                input_event = int(input_event)
                input_event = self.id2token[input_event]
            except:
                raise TypeError("input_event should be int or str, input_event={}, type={}".format(input_event,type(input_event)))
        
        if not input_event.startswith("Note-On"):
            return -1

        # prefix = input_event.split("_")[0]
        # base = self.token_type_base[prefix]
        # pitch = int(input_event.split("_")[1])
        # return self._pitch_bins[pitch - base]
        assert int(input_event.split("_")[1]) >=1 and int(input_event.split("_")[1]) <=127
        return int(input_event.split("_")[1])
    def getTrackName(self,input_event):
        """Return corresponding note track name
        if input_event is not a note, it returns -1

        Args:
            input_event (str or int): REMI Event Name or vocab ID
        """
        if isinstance(input_event,int):
            input_event = self.id2token[input_event]
        elif isinstance(input_event,str):
            pass
        else:
            try:
                input_event = int(input_event)
                input_event = self.id2token[input_event]
            except:
                raise TypeError("input_event should be int or str, input_event={}, type={}".format(input_event,type(input_event)))
        
        if not input_event.startswith("Note-On"):
            return -1

        # prefix = input_event.split("_")[0]
        # base = self.token_type_base[prefix]
        # pitch = int(input_event.split("_")[1])
        # return self._pitch_bins[pitch - base]
        input_event = input_event.replace("Note-On-","")
        input_event = "_".join(input_event.split("_")[:-1])
        assert input_event in self.track_names , input_event

        return input_event
        
    def parseMidi(self,midi_path,verbose=False):
        midi_obj: miditoolkit.midi.MidiFile = mid_parser.MidiFile(midi_path)
        output_tracks = []
        assert midi_obj.time_signature_changes[0].denominator == 4
        assert midi_obj.time_signature_changes[0].numerator == 4
        bar_edges = np.arange(start=0,stop = midi_obj.max_tick,step=midi_obj.ticks_per_beat*4)
        ticks_per_bar = midi_obj.ticks_per_beat * 4
        ticks_per_step = midi_obj.ticks_per_beat / self.q_beat
        all_events_grps =  []
        for i in range(len(bar_edges)-1): all_events_grps.append([])
        tempo_id = (np.abs(self._tempo_bins - midi_obj.tempo_changes[0].tempo)).argmin()
        
        # identify melody track
        print(midi_path)
        melody_track_id,melody_track_name  = self.getMelodyTrack(midi_obj)
        if melody_track_id is None:
            print("Skip files with no melody")
            return None
        
        # do not use piano as melody
        if melody_track_name.lower() == "piano": 
            print("Skip files with piano as melody")
            return None
        
        # collect tracks' note
        for midi_track_i, midi_track in enumerate(midi_obj.instruments):
            assert midi_track.name in self.track_names

            _track_name = "Melody" if midi_track_i == melody_track_id else midi_track.name
            for bar_index, bar_start_tick in enumerate(bar_edges[:-1]):
                if verbose:
                    print("Track:{} Bar #{}, start at tick:{}".format(_track_name,bar_index,bar_start_tick))

                notes = list(filter(lambda n: n.start >= bar_start_tick and n.start < bar_start_tick + ticks_per_bar and n.pitch > 0,midi_track.notes))
                notes = sorted(notes,key= lambda n: (n.start,-n.pitch))
                for n in notes:
                    if verbose:
                        print(n)
                    pos = int((n.start - bar_start_tick)  // ticks_per_step)
                    assert not (pos == 16)
                    all_events_grps[bar_index].append({
                        "type" : "note",
                        "pos" : pos,
                        "pitch" : n.pitch,
                        "dur" : min(int((n.end-n.start)//ticks_per_step),len(self._duration_bins)-1),
                        "vel" : n.velocity,
                        "track_name" : _track_name,
                        "priority" : 1 + self.track_names.index(_track_name)
                    })
        midi_obj.markers = sorted(midi_obj.markers,key=lambda x: x.time)
        # # print(midi_obj.lyrics)
        # print(midi_obj.markers[-1])
        # print(bar_edges[-2:])
        # print(len(midi_obj.markers))
        # print(len(bar_edges))
        # exit()
        # collect chords
        for bar_index, bar_start_tick in enumerate(bar_edges[:-1]):
            if verbose:
                print("Chord Bar #{}, start at tick:{}".format(bar_index,bar_start_tick))
            chords = list(filter(lambda n: n.time >= bar_start_tick and n.time < bar_start_tick + ticks_per_bar,midi_obj.markers))
            for _chord_i,_chord in enumerate(chords):
                pos = int((_chord.time - bar_start_tick) // ticks_per_step)
                if not _chord.text.split('_')[0] == 'N':
                    all_events_grps[bar_index].append({
                        "type" : "chord",
                        "pos" : pos,
                        "root" : _chord.text.split('_')[0],
                        "quality" : _chord.text.split('_')[1],
                        "priority" : 0,
                        "pitch" :0
                    })
            # notes = sorted(notes,key= lambda n: (n.start,-n.pitch))

        all_words = []
        all_words.append("Piece-Start")
        all_words.append("Tempo_{}".format(self._tempo_bins[tempo_id]))
        # convert to events
        # for bar_index, bar_start_tick in enumerate(bar_edges[:-1]):
        for bar_index, grp in enumerate(all_events_grps):
            if verbose:
                print("Bar #{}".format(bar_index))
            last_pos = -1
            last_track_name = ""
            all_words.append("Bar-Start")
            grp = sorted(grp,key= lambda n: (n["pos"],n["priority"],n["pitch"]))
            for n in grp:
                if verbose:
                    print(n)
                pos = n["pos"]
                if not pos == last_pos:
                    all_words.append("Position_{}".format(pos))
                    last_pos = pos
                    last_track_name = ""
                if n["type"] == "note":
                    # if not last_track_name == n["track_name"]:
                    #     last_track_name = n["track_name"]
                    #     all_words.append("Track_{}".format(n["track_name"]))
                    
                    all_words.append("Note-On-{}_{}".format(n["track_name"], n["pitch"]))
                    all_words.append("Note-Duration_{}".format(n["dur"]))
                    all_words.append("Note-Velocity_{}".format(n["vel"]))
                elif n["type"] == "chord":
                    all_words.append("Chord-Root_{}".format(n["root"]))
                    all_words.append("Chord-Quality_{}".format(n["quality"]))
                else:
                    print("Error item: {}".format(n))
                    exit(1)

        all_words.append("Piece-End")

        all_ids = [ self.token2id[w] for w in all_words]

        return all_ids

    def toMidi(self,midi_path,tokenIDs):
        newMidiObj = miditoolkit.midi.MidiFile()
        ticks_per_bar = newMidiObj.ticks_per_beat * 4
        ticks_per_step = newMidiObj.ticks_per_beat / self.q_beat
        
        all_instruments = {
            _inst_name : ct.Instrument(
                program=grp2Inst[_inst_name] if not _inst_name=="Drum" else 0,
                is_drum=(_inst_name=="Drum"),
                name=_inst_name
            ) for _inst_name in self.track_names
        }

        all_tokens = [ self.id2token[w] for w in tokenIDs]
        assert all_tokens[0] == "Piece-Start"
        assert all_tokens[1].startswith("Tempo")
        tempo = int(all_tokens[1].split("_")[1])
        newMidiObj.tempo_changes.append(ct.TempoChange(tempo = tempo,time = 0))
        all_tokens = all_tokens[2:]

        i = 0
        current_bar_start = - ticks_per_bar
        current_pos = 0
        last_track_name = ""
        while(i < len(all_tokens)):
            tok = all_tokens[i]
            if tok.startswith("Bar"):
                if tok == "Bar-Start":
                    current_bar_start += ticks_per_bar
                i += 1
            elif tok.startswith("Position"):
                current_pos = int(tok.split("_")[-1])
                last_track_name = ""
                i += 1
            elif tok.startswith("Note-On"):
                # print(i,tok)
                if not all_tokens[i+1].startswith("Note-Duration"):
                    i += 1
                    print("Skip {} becasue {}".format(tok,all_tokens[i+1]))
                    continue
                if not all_tokens[i+2].startswith("Note-Velocity"):
                    i += 2
                    print("Skip {},{} becasue {}".format(tok,all_tokens[i+1],all_tokens[i+2]))
                    continue
                assert all_tokens[i+1].startswith("Note-Duration")
                assert all_tokens[i+2].startswith("Note-Velocity")
                
                _track_name = self.getTrackName(tok)
                assert _track_name in self.track_names
                
                pitch = int(tok.split("_")[-1])
                dur = int(all_tokens[i+1].split("_")[-1]) * ticks_per_step
                vec = int(all_tokens[i+2].split("_")[-1])
                all_instruments[_track_name].notes.append(miditoolkit.midi.containers.Note(
                    velocity=vec,
                    pitch=pitch,
                    start=int(current_pos*ticks_per_step+current_bar_start),
                    end=int(current_pos*ticks_per_step+current_bar_start+dur)
                ))
                i += 3
            elif tok.startswith("Chord-Root"):
                # assert all_tokens[i+1].startswith("Chord-Quality")
                if not all_tokens[i+1].startswith("Chord-Quality"):
                    print("No Chord-Quality event specified, skipped {}".format(tok))
                    i += 1
                    continue
                newMidiObj.markers.append(
                    ct.Marker(text="{}_{}".format(tok.replace("Chord-Root_",""),tok.replace("Chord-Quality_","")),
                        time=int(current_pos*ticks_per_step+current_bar_start)
                        )
                )
                i+=2
            elif tok.startswith("Piece-End"):
                break
            else:
                print("Error tok:{}, skipped".format(tok))
                i+=1
            
        newMidiObj.instruments = [ all_instruments[inst_name] for inst_name in all_instruments]

        newMidiObj.dump(midi_path)

    def sampleFromTracks(self,tracks):
        _track_ids = [x for x in range(len(tracks))]
        random.shuffle(_track_ids)
        _out = []
        _out.append(self.token2id["Piece-Start"])
        for _tId in _track_ids:
            _out.append(self.token2id["Track-Start"])
            _out.extend(tracks[_tId])
            _out.append(self.token2id["Track-End"])
        _out.append(self.token2id["Piece-End"])

        return _out

    def dumpToText(self,seqIDs: list, delimeter: str="\n") -> str:
        _s: list = [ self.id2token[x] for x in seqIDs]
        return delimeter.join(_s)

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

class Vocab_wMelody(object):
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

        # split each beat into self.q_bar subbeats
        self.q_beat: int = 4

        # midi pitch 0 (padding) ,1 ~ 127(highest pitch) 
        self._pitch_bins: np.ndarray = np.arange(start=1,stop=128)
        
        # midi program number
        # self._inst_bins: np.ndarray = np.arange(start=0,stop=128)

        # about 98% of the notes in LMD <= 1 bar length
        self._duration_bins = np.arange(start=1,stop=self.q_beat*4+1)

        self._velocity_bins =  np.arange(start=1,stop=127+1)

        self._tempo_bins = np.arange(start=30,stop=197,step=3)

        self._position_bins = np.arange(start=0,stop=self.q_beat*4)

        # track group
        self.track_names = ["Drum","Melody","Piano","Ensemble","Bass","Guitar","Brass","Pipe","Reed","Synth_Lead"]

        # chords
        self.chord_roots = list(Vocab.pitch2num.keys())
        self.chord_qualities = chorder.Chord.qualities


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

        
        # Note-On
        self.token_type_base = {'Note-On' : 1}
        for i in self._pitch_bins:
            self.token2id[ 'Note-On_{}'.format(i) ] = self.n_tokens
            self.n_tokens += 1
        
        

        # Note-Duration
        self.token_type_base['Note-Duration'] = self.n_tokens
        for note_dur in self._duration_bins:
            self.token2id[ 'Note-Duration_{}'.format(note_dur) ] = self.n_tokens
            self.n_tokens += 1


        # Note-Velocity
        self.token_type_base['Note-Velocity'] = self.n_tokens
        for vel in self._velocity_bins:
            self.token2id[ 'Note-Velocity_{}'.format(vel) ] = self.n_tokens
            self.n_tokens += 1
        
        # Tempo
        self.token_type_base['Tempo'] = self.n_tokens
        for tmp in self._tempo_bins:
            self.token2id[ 'Tempo_{}'.format(tmp) ] = self.n_tokens
            self.n_tokens += 1

        # Positions
        self.token_type_base['Position'] = self.n_tokens
        for pos in self._position_bins:
            self.token2id[ 'Position_{}'.format(pos) ] = self.n_tokens
            self.n_tokens += 1

        # bar start and end
        self.token_type_base['Bar'] = self.n_tokens
        self.token2id[ 'Bar-Start' ] = self.n_tokens
        self.n_tokens += 1
        # self.token_type_base['Bar'] = self.n_tokens
        # self.token2id[ 'Bar-End' ] = self.n_tokens
        # self.n_tokens += 1

        # # track start and end
        # self.token_type_base['Track'] = self.n_tokens
        # self.token2id[ 'Track-Start' ] = self.n_tokens
        # self.n_tokens += 1
        # self.token_type_base['Track'] = self.n_tokens
        # self.token2id[ 'Track-End' ] = self.n_tokens
        # self.n_tokens += 1

        # piece start and end
        self.token_type_base['Piece'] = self.n_tokens
        self.token2id[ 'Piece-Start' ] = self.n_tokens
        self.n_tokens += 1
        self.token_type_base['Piece'] = self.n_tokens
        self.token2id[ 'Piece-End' ] = self.n_tokens
        self.n_tokens += 1
        
        # track names
        self.token_type_base['Track'] = self.n_tokens
        for t in self.track_names:
            self.token2id[ 'Track_{}'.format(t) ] = self.n_tokens
            self.n_tokens += 1
        
        self.n_tokens = len(self.token2id)

        # chord events
        self.token_type_base['Chord'] = self.n_tokens
        for t in self.chord_roots:
            self.token2id[ 'Chord-Root_{}'.format(t) ] = self.n_tokens
            self.n_tokens += 1
        
        for t in self.chord_qualities:
            self.token2id[ 'Chord-Quality_{}'.format(t) ] = self.n_tokens
            self.n_tokens += 1
        
        self.n_tokens = len(self.token2id)

        for w , v in self.token2id.items():
            self.id2token[v] = w
            
    def getMelodyTrack(self,midiobj:miditoolkit.MidiFile):
        assert len(midiobj.lyrics) > 0, "lyrics should exists"
        timestep = int(midiobj.ticks_per_beat // 4)
        melody_onset_times = [ int( round(n.time / timestep)*timestep )  for n in midiobj.lyrics]
        intersect_scores = []

        ins_names = [x.name for x in midiobj.instruments ]
        for ins in midiobj.instruments:
            if ins.is_drum is True: 
                intersect_scores.append(0)
                continue
            # _ins_onset_times = [ int( round(n.start / timestep)*timestep ) for n in ins.notes]
            _ins_onset_times = [ (n.start  ) for n in ins.notes]
            intersect_scores.append( len(compare_intersect(melody_onset_times,_ins_onset_times)) / len(melody_onset_times))
        # print(intersect_scores)
        # print(ins_names)
        best_match_id = np.argmax(intersect_scores)
        # print("best_match_id",best_match_id)

        if intersect_scores[best_match_id] < 0.6:
            print(intersect_scores)
            print(ins_names)
            return None, None
        assert intersect_scores[best_match_id] >= 0.6, intersect_scores
        return best_match_id, ins_names[best_match_id]
        

    def getPitch(self,input_event):
        """Return corresponding note pitch
        if input_event is not a note, it returns -1

        Args:
            input_event (str or int): REMI Event Name or vocab ID
        """
        if isinstance(input_event,int):
            input_event = self.id2token[input_event]
        elif isinstance(input_event,str):
            pass
        else:
            try:
                input_event = int(input_event)
                input_event = self.id2token[input_event]
            except:
                raise TypeError("input_event should be int or str, input_event={}, type={}".format(input_event,type(input_event)))
        
        if not input_event.startswith("Note-On"):
            return -1

        # prefix = input_event.split("_")[0]
        # base = self.token_type_base[prefix]
        # pitch = int(input_event.split("_")[1])
        # return self._pitch_bins[pitch - base]
        assert int(input_event.split("_")[1]) >=1 and int(input_event.split("_")[1]) <=127
        return int(input_event.split("_")[1])
      
    def parseMidi(self,midi_path,verbose=False):
        midi_obj: miditoolkit.midi.MidiFile = mid_parser.MidiFile(midi_path)
        output_tracks = []
        assert midi_obj.time_signature_changes[0].denominator == 4
        assert midi_obj.time_signature_changes[0].numerator == 4
        bar_edges = np.arange(start=0,stop = midi_obj.max_tick,step=midi_obj.ticks_per_beat*4)
        ticks_per_bar = midi_obj.ticks_per_beat * 4
        ticks_per_step = midi_obj.ticks_per_beat / self.q_beat
        all_events_grps =  []
        for i in range(len(bar_edges)-1): all_events_grps.append([])
        tempo_id = (np.abs(self._tempo_bins - midi_obj.tempo_changes[0].tempo)).argmin()
        
        # identify melody track
        print(midi_path)
        melody_track_id,melody_track_name  = self.getMelodyTrack(midi_obj)
        if melody_track_id is None:
            print("Skip files with no melody")
            return None
        
        # do not use piano as melody
        if melody_track_name.lower() == "piano": 
            print("Skip files with piano as melody")
            return None
        
        # collect tracks' note
        for midi_track_i, midi_track in enumerate(midi_obj.instruments):
            assert midi_track.name in self.track_names

            _track_name = "Melody" if midi_track_i == melody_track_id else midi_track.name
            for bar_index, bar_start_tick in enumerate(bar_edges[:-1]):
                if verbose:
                    print("Track:{} Bar #{}, start at tick:{}".format(_track_name,bar_index,bar_start_tick))

                notes = list(filter(lambda n: n.start >= bar_start_tick and n.start < bar_start_tick + ticks_per_bar and n.pitch > 0,midi_track.notes))
                notes = sorted(notes,key= lambda n: (n.start,-n.pitch))
                for n in notes:
                    if verbose:
                        print(n)
                    pos = int((n.start - bar_start_tick)  // ticks_per_step)
                    assert not (pos == 16)
                    all_events_grps[bar_index].append({
                        "type" : "note",
                        "pos" : pos,
                        "pitch" : n.pitch,
                        "dur" : min(int((n.end-n.start)//ticks_per_step),len(self._duration_bins)-1),
                        "vel" : n.velocity,
                        "track_name" : _track_name,
                        "priority" : 1 + self.track_names.index(_track_name)
                    })
        midi_obj.markers = sorted(midi_obj.markers,key=lambda x: x.time)
        # # print(midi_obj.lyrics)
        # print(midi_obj.markers[-1])
        # print(bar_edges[-2:])
        # print(len(midi_obj.markers))
        # print(len(bar_edges))
        # exit()
        # collect chords
        for bar_index, bar_start_tick in enumerate(bar_edges[:-1]):
            if verbose:
                print("Chord Bar #{}, start at tick:{}".format(bar_index,bar_start_tick))
            chords = list(filter(lambda n: n.time >= bar_start_tick and n.time < bar_start_tick + ticks_per_bar,midi_obj.markers))
            for _chord_i,_chord in enumerate(chords):
                pos = int((_chord.time - bar_start_tick) // ticks_per_step)
                if not _chord.text.split('_')[0] == 'N':
                    all_events_grps[bar_index].append({
                        "type" : "chord",
                        "pos" : pos,
                        "root" : _chord.text.split('_')[0],
                        "quality" : _chord.text.split('_')[1],
                        "priority" : 0,
                        "pitch" :0
                    })
            # notes = sorted(notes,key= lambda n: (n.start,-n.pitch))

        all_words = []
        all_words.append("Piece-Start")
        all_words.append("Tempo_{}".format(self._tempo_bins[tempo_id]))
        # convert to events
        # for bar_index, bar_start_tick in enumerate(bar_edges[:-1]):
        for bar_index, grp in enumerate(all_events_grps):
            if verbose:
                print("Bar #{}".format(bar_index))
            last_pos = -1
            last_track_name = ""
            all_words.append("Bar-Start")
            grp = sorted(grp,key= lambda n: (n["pos"],n["priority"],n["pitch"]))
            for n in grp:
                if verbose:
                    print(n)
                pos = n["pos"]
                if not pos == last_pos:
                    all_words.append("Position_{}".format(pos))
                    last_pos = pos
                    last_track_name = ""
                if n["type"] == "note":
                    if not last_track_name == n["track_name"]:
                        last_track_name = n["track_name"]
                        all_words.append("Track_{}".format(n["track_name"]))
                    
                    all_words.append("Note-On_{}".format(n["pitch"]))
                    all_words.append("Note-Duration_{}".format(n["dur"]))
                    all_words.append("Note-Velocity_{}".format(n["vel"]))
                elif n["type"] == "chord":
                    all_words.append("Chord-Root_{}".format(n["root"]))
                    all_words.append("Chord-Quality_{}".format(n["quality"]))
                else:
                    print("Error item: {}".format(n))
                    exit(1)

        all_words.append("Piece-End")

        all_ids = [ self.token2id[w] for w in all_words]

        return all_ids

    def toMidi(self,midi_path,tokenIDs):
        newMidiObj = miditoolkit.midi.MidiFile()
        ticks_per_bar = newMidiObj.ticks_per_beat * 4
        ticks_per_step = newMidiObj.ticks_per_beat / self.q_beat
        
        all_instruments = {
            _inst_name : ct.Instrument(
                program=grp2Inst[_inst_name] if not _inst_name=="Drum" else 0,
                is_drum=(_inst_name=="Drum"),
                name=_inst_name
            ) for _inst_name in self.track_names
        }

        all_tokens = [ self.id2token[w] for w in tokenIDs]
        assert all_tokens[0] == "Piece-Start"
        assert all_tokens[1].startswith("Tempo")
        tempo = int(all_tokens[1].split("_")[1])
        newMidiObj.tempo_changes.append(ct.TempoChange(tempo = tempo,time = 0))
        all_tokens = all_tokens[2:]

        i = 0
        current_bar_start = - ticks_per_bar
        current_pos = 0
        last_track_name = ""
        while(i < len(all_tokens)):
            tok = all_tokens[i]
            if tok.startswith("Bar"):
                if tok == "Bar-Start":
                    current_bar_start += ticks_per_bar
                i += 1
            elif tok.startswith("Position"):
                current_pos = int(tok.split("_")[-1])
                last_track_name = ""
                i += 1
            elif tok.startswith("Track"):
                last_track_name = tok[6:]
                i += 1
            elif tok.startswith("Note-On"):
                # print(i,tok)
                if not all_tokens[i+1].startswith("Note-Duration"):
                    i += 1
                    print("Skip {} becasue {}".format(tok,all_tokens[i+1]))
                    continue
                if not all_tokens[i+2].startswith("Note-Velocity"):
                    i += 2
                    print("Skip {},{} becasue {}".format(tok,all_tokens[i+1],all_tokens[i+2]))
                    continue
                assert all_tokens[i+1].startswith("Note-Duration")
                assert all_tokens[i+2].startswith("Note-Velocity")
                if last_track_name == "":
                    print("No track event specified, skipped {} {} {}".format(tok,all_tokens[i+1],all_tokens[i+2]))
                    i += 3
                    continue
                
                pitch = int(tok.split("_")[-1])
                dur = int(all_tokens[i+1].split("_")[-1]) * ticks_per_step
                vec = int(all_tokens[i+2].split("_")[-1])
                all_instruments[last_track_name].notes.append(miditoolkit.midi.containers.Note(
                    velocity=vec,
                    pitch=pitch,
                    start=int(current_pos*ticks_per_step+current_bar_start),
                    end=int(current_pos*ticks_per_step+current_bar_start+dur)
                ))
                i += 3
            elif tok.startswith("Chord-Root"):
                # assert all_tokens[i+1].startswith("Chord-Quality")
                if not all_tokens[i+1].startswith("Chord-Quality"):
                    print("No Chord-Quality event specified, skipped {}".format(tok))
                    i += 1
                    continue
                newMidiObj.markers.append(
                    ct.Marker(text="{}_{}".format(tok.replace("Chord-Root_",""),tok.replace("Chord-Quality_","")),
                        time=int(current_pos*ticks_per_step+current_bar_start)
                        )
                )
                i+=2
            elif tok.startswith("Piece-End"):
                break
            else:
                print("Error tok:{}, skipped".format(tok))
                i+=1
            
        newMidiObj.instruments = [ all_instruments[inst_name] for inst_name in all_instruments]

        newMidiObj.dump(midi_path)

    def sampleFromTracks(self,tracks):
        _track_ids = [x for x in range(len(tracks))]
        random.shuffle(_track_ids)
        _out = []
        _out.append(self.token2id["Piece-Start"])
        for _tId in _track_ids:
            _out.append(self.token2id["Track-Start"])
            _out.extend(tracks[_tId])
            _out.append(self.token2id["Track-End"])
        _out.append(self.token2id["Piece-End"])

        return _out

    def dumpToText(self,seqIDs: list, delimeter: str="\n") -> str:
        _s: list = [ self.id2token[x] for x in seqIDs]
        return delimeter.join(_s)

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
    # myvocab = Vocab_wMelody()
    myvocab = Vocab_wMelody_DenseTrack()

    # print(myvocab)
    # print(len(myvocab))
    t = myvocab.parseMidi("../../mydataset/LMDfull_dataset_midi_with_chords_selected/lmd_10022_full.mid",False)
    # print(myvocab.dumpToText(t))
    # myvocab.toMidi("asd.mid",tokenIDs=t)
    
    exit()
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


