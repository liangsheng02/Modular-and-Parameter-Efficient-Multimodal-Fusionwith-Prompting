import re, tqdm, json
from typing import Sequence
import torch
import io
import pyarrow as pa
import os
from PIL import Image
import librosa


class MustardCollator:
    def __init__(self, a_feature_extractor, v_feature_extractor, tokenizer, blacked=False):
        self.a_feature_extractor = a_feature_extractor
        self.v_feature_extractor = v_feature_extractor
        self.tokenizer = tokenizer
        self.blacked = blacked

    def __call__(self, batch):
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        new_dict_batch = {}

        img_key = "frames"
        img_list = list(map(list, zip(*dict_batch[img_key])))
        new_img_list = []
        for i in range(img_list.__len__()):
            new_img_list.append(self.v_feature_extractor(img_list[i], padding=True, return_tensors="pt")['pixel_values'])
        new_dict_batch["vis_feas"] = None if self.blacked else new_img_list   # a list of 20 tensors with shape [8,3,224,224]

        wav_key = "wav"
        new_dict_batch["aud_feas"] = None if self.blacked else self.a_feature_extractor(dict_batch[wav_key],
                                                                                        padding=True,
                                                                                        sampling_rate=16000,
                                                                                        return_tensors="pt")['input_values']
        txt_key = "text"
        new_text = self.tokenizer(dict_batch[txt_key], padding=True, return_tensors="pt")
        new_dict_batch["input_ids"] = new_text.input_ids
        new_dict_batch["attention_mask"] = new_text.attention_mask

        label_key = "labels"
        if label_key in keys:
            # label = ["Is irony:"+str(t) for t in dict_batch[label_key]]
            label = [str(t) for t in dict_batch[label_key]]
            new_label = self.tokenizer(label, padding=True, return_tensors="pt").input_ids
            new_dict_batch["labels"] = new_label
        return new_dict_batch


class MustardDataset(torch.utils.data.Dataset):
    def __init__(self, *args, arrow_path="", **kwargs):
        super().__init__(*args, **kwargs)
        self.table = pa.ipc.RecordBatchFileReader(pa.memory_map(arrow_path, "r")).read_all()

    def __len__(self):
        return self.table.num_rows

    def __getitem__(self, index):
        id = self.table["id"][index].as_py()
        text = self.table["utterance"][index].as_py()
        labels = self.table["sarcasm"][index].as_py()
        frames = [Image.open(f).convert("RGB") for f in self.table['frames'][index].as_py()]
        wav, _ = librosa.load(self.table['wav'][index].as_py(), sr=16000)

        return {
            "id": id,
            "text": text,
            "labels": labels,
            "frames": frames,
            "wav": wav,
        }
