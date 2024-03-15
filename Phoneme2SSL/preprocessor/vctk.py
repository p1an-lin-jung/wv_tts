import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]

    with open(os.path.join(in_dir, "content.txt"), encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            wav_path, text,speaker = line.strip("\n").split("|")
            
            text = text.split(" ")

            if os.path.exists(wav_path):
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                wav, _ = librosa.load(wav_path, sampling_rate)
                wav = wav / max(abs(wav)) * max_wav_value
                wavfile.write(
                    os.path.join(out_dir, speaker, wav_path),
                        sampling_rate,
                        wav.astype(np.int16),
                    )
                with open(
                        os.path.join(out_dir, speaker, "{}.lab".format(wav_path[:-4])),
                        "w",
                ) as f1:
                    f1.write(" ".join(text))