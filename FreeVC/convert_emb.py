import os
import argparse
import torch
import librosa
import time
from scipy.io.wavfile import write
from tqdm import tqdm
import numpy as np
import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from wavlm import WavLM, WavLMConfig
from speaker_encoder.voice_encoder import SpeakerEncoder
import logging
logging.getLogger('numba').setLevel(logging.WARNING)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", type=str, default="configs/freevc-nnemb.json", help="path to json config file")
    parser.add_argument("--ptfile", type=str, default="logs/freevc-nosr_vctk/G_160000.pth", help="path to pth file")
    
    parser.add_argument("--txtpath", type=str, default="./filelists/convert.txt", help="path to txt file")
    parser.add_argument("--outdir", type=str, default="output/vctk/", help="path to output dir")
    parser.add_argument("--use_timestamp", default=False, action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    hps = utils.get_hparams_from_file(args.hpfile)

    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()

    _ = net_g.eval()
    print("Loading checkpoint...")
    _ = utils.load_checkpoint(args.ptfile, net_g, None, True)
    
    if not hps.model.use_spk:
        print("Loading speaker encoder...")
        smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    print("Processing text...")
    titles, srcs, tgts = [], [], []
    with open(args.txtpath, "r") as f:
        for rawline in f.readlines():
            title, src, tgt = rawline.strip().split("|")
            titles.append(title)
            srcs.append(src)
            tgts.append(tgt)
    import json
    with open(os.path.join('filelists', "speakers.json")) as f:
        speaker_map = json.load(f)

    print("Synthesizing...")
    with torch.no_grad():
        for line in tqdm(zip(titles, srcs, tgts)):
            title, src, spk = line
            # spekaer_id= int(spk)
            
            spekaer_id=speaker_map[spk]
            speakers = np.array(speakers)
        
            speakers = torch.from_numpy(np.array(spekaer_id)).long().unsqueeze(0).cuda()
            
            c = np.load(src)
            c = torch.from_numpy(c)
            c = c.transpose(1, 2)
            c= c.cuda()
            if hps.model.use_spk:
                audio = net_g.infer(c,)
            else:
                audio = net_g.infer(c, speakers)
            audio = audio[0][0].data.cpu().float().numpy()
            if args.use_timestamp:
                timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
                write(os.path.join(args.outdir, "{}.wav".format(timestamp+"_"+title)), hps.data.sampling_rate, audio)
            else:
                write(os.path.join(args.outdir, f"{title}.wav"), hps.data.sampling_rate, audio)
            