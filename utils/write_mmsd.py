import json
import pandas as pd
import pyarrow as pa
import random
import os


def make_arrow(root, arrow_root):

    with open(f"{root}/sarcasm_data.json", "r") as fp:
        annotations = json.load(fp)

    out = {"full": [], "speaker_d_train": [], "speaker_d_test": []}
    for id in annotations.keys():
        # Video
        v_list = os.listdir(f"{root}/mustard_v/{id}")
        v_len = v_list.__len__()
        if v_len >= 0:
            # Frames
            frames = [f"{root}/mustard_v/{id}/{v_list[i]}"
                      for i in list(range(0, v_list.__len__()))]
            # Text
            show = annotations[id]['show']
            sarcasm = annotations[id]['sarcasm']
            utterance = annotations[id]['utterance']
            # Audio
            wav = f"{root}/mustard_a/{id}.wav"
            out['full'].append([id, show, utterance, sarcasm, frames, wav])
            if show == 'FRIENDS':
                out['speaker_d_test'].append([id, show, utterance, sarcasm, frames, wav])
            else:
                out['speaker_d_train'].append([id, show, utterance, sarcasm, frames, wav])
            # print(len(out['speaker_d_test']), len(out['speaker_d_train']))
    
    for split in out.keys():
        dataframe = pd.DataFrame(out[split], columns=["id", "show", "utterance", "sarcasm", "frames", "wav", ],)
        table = pa.Table.from_pandas(dataframe)
    
        with pa.OSFile(f"{arrow_root}/mustard_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        print(f"{arrow_root}/mustard_{split}.arrow")
