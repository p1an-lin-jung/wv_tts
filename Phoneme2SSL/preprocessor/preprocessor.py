import os
import random
import json

import tgt
import librosa
import numpy as np
from tqdm import tqdm
import torch
import pdb
import audio as Audio
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_wav2vec_large(model_path):
    import transformers
    cmodel = transformers.Wav2Vec2ForPreTraining.from_pretrained(model_path)
    for param in cmodel.parameters():
        param.requires_grad = False
        param.grad = None
    cmodel.eval()
    cmodel.to(device)
    print("Loaded wav2vec2.0.")
    return cmodel 

def get_w2_feat_en(wav_path,cmodel,start=0,end=0):
    wav, sr = librosa.load(wav_path,sr=16000)
    if start<end:
        wav = wav[
        int(sr * start) : int(sr * end)
        ].astype(np.float32)
    wav = torch.from_numpy(wav).unsqueeze(0).cuda()
    with torch.no_grad():
        c = cmodel(wav, output_hidden_states=True)
    return c.hidden_states[-1].cpu().numpy()

def seconds_to_w2vframe(sample_point):
    conv_dim= [512,512,512,512,512,512,512]
    conv_kernel= [10,3,3,3,3,2,2]
    conv_stride= [5,2,2,2,2,2,2]

    out=sample_point
    for i in range(len(conv_dim)):
        out=(out-conv_kernel[i])//conv_stride[i]+1
    return out



class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.ssl_sampling_rate = config["preprocessing"]["w2v"]["sampling_rate"]


        self.model=get_model_wav2vec_large(config["path"]["wav2vec2_path"])

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "ssl")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "ssl_duration")), exist_ok=True)
        print("Processing Data ...")
        out = list()

        speakers = {}
        # fw=open('./preprocessed_data/VCTK/miss.txt','w',encoding='utf-8')
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            speakers[speaker] = i
            print(speaker)
            for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
                if ".wav" not in wav_name:
                    continue
                basename = wav_name.split(".")[0]
                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                )

                if os.path.exists(tg_path):
                    ret = self.process_utterance(speaker, basename)
                    print(ret)
                    if ret is None:
                        continue
                    else:
                        info = ret
                    out.append(info)
                else:
                    # print("./raw_data/VCTK/{}/{}".format(speaker,wav_name),file=fw)
                    continue

        # fw.close()
        print("Computing statistic quantities ...")
        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        random.shuffle(out)
        out = [r for r in out if r is not None]
        return out

    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, ssl_durations, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        wv_feat= get_w2_feat_en(wav_path,self.model,start,end)

        # Due to the rounding operation, 
        # the converted frame length may differ from the ground-truth frame length by one.
        if (wv_feat.shape[1]<sum(ssl_durations)):
            max_index = np.argmax(ssl_durations)
            ssl_durations[max_index]-=1

        wv_feat =wv_feat[:sum(ssl_durations)]
    
        ssl_dur_filename = "{}-ssl_duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "ssl_duration", ssl_dur_filename), ssl_durations)

        ssl_filename = "{}-ssl-{}.npy".format(speaker, basename) 
        np.save(
            os.path.join(self.out_dir, "ssl", ssl_filename),
            wv_feat,
        )    
        wv_feat_path=os.path.join(self.out_dir, "ssl", ssl_filename)
 
        return (
            "|".join([basename, speaker, text, raw_text,wv_feat_path])
        )
    
    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]
        phones = []
        durations = []
        ssl_durations=[]
        start_time = 0
        end_time = 0
        end_idx = 0

        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            ssl_durations.append(
                int(
                    np.round(seconds_to_w2vframe(e * self.ssl_sampling_rate))
                    - np.round(seconds_to_w2vframe(s * self.ssl_sampling_rate))
                )
            )
        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]
        ssl_durations = ssl_durations[:end_idx]
    
        return phones, ssl_durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value