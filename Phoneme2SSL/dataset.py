import json
import math
import os
import torch
import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.raw_path = preprocess_config["path"]["raw_path"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
 

        self.basename, self.speaker, self.text, self.raw_text, self.ssl_emb_paths = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last
 
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        ssl_emb_path = self.ssl_emb_paths[idx]

 
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
   
        ssl_duration_path = os.path.join(
            self.preprocessed_path,
            "ssl_duration",
            "{}-ssl_duration-{}.npy".format(speaker, basename),
        )
        ssl_duration = np.load(ssl_duration_path)
        ssl_emb=np.load(ssl_emb_path)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "ssl_emb":ssl_emb[0],
            "ssl_duration":ssl_duration,            
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text_ph = []
            raw_text = []
            ssl_emb= []
            for line in f.readlines():

                n, s, tp, r, ssl = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text_ph.append(tp)
                raw_text.append(r)
                ssl_emb.append(ssl)
            return name, speaker, text_ph, raw_text,ssl_emb


    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        ssl_embs= [data[idx]["ssl_emb"] for idx in idxs]
        ssl_durations=[data[idx]["ssl_duration"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        ssl_lens = np.array([ssl_emb.shape[0] for ssl_emb in ssl_embs])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        ssl_durations= pad_1D(ssl_durations)
        ssl_embs = pad_2D(ssl_embs)
        return (
            ids,
            raw_texts,
            ssl_embs,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            ssl_lens,
            max(ssl_lens),
            ssl_durations,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        self.basename, self.speaker, self.text, self.raw_text,self.ssl_path = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        ssl=self.ssl_path[idx]
        return (basename, speaker_id, phone, raw_text, ssl)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            ssl_emb=[]
            for line in f.readlines()[:50]:
                n, s, tp, r, ssl = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(tp)
                raw_text.append(r)
                ssl_emb.append(ssl)
            return name, speaker, text, raw_text,ssl_emb

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        ssl_paths = [d[4] for d in data]
        ssl_lens=[]
        for ssl_path in ssl_paths:
            ssl_emb=np.load(ssl_path)
            ssl_lens.append(ssl_emb.shape[1])
        text_lens = np.array([text.shape[0] for text in texts])
        texts = pad_1D(texts)
        return ids, raw_texts, speakers, texts, text_lens, max(text_lens),ssl_lens,ssl_paths
