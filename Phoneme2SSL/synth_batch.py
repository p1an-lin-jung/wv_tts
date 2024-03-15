import re
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style
import pdb

from utils.tools import to_device
from dataset import TextDataset
from text import text_to_sequence
import os
from model import Phoneme2SSL
from utils.model import get_model, get_param_num

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def infer_batch(model,  batchs, control_values,target_dir):
    duration_control = control_values
    os.makedirs('results/'+target_dir,exist_ok=True)

    output_paths=[]
    preds=[]
    gts=[]
    ssl_paths=[]
    basenames=[]
    spks=[]
    for batch in batchs:
        batch = to_device(batch, device)

        with torch.no_grad():

            output = model(
                *(batch[2:-1]),
                d_control=duration_control
            )
        feat=output[0]
        output_path='results/'+target_dir+'/'+batch[0][0]+"_feat.npy"
        
        np.save(output_path,feat.cpu().numpy())
        output_paths.append(output_path)
        preds.append(feat.shape[1])
        gts.append(batch[6][0])
        ssl_paths.append(batch[7][0])
        basenames.append(batch[0][0])
        spks.append(batch[2][0])
    return preds,gts,output_paths,ssl_paths,basenames,spks

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--target_dir",type=str,default='vctk_result'
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    parser.add_argument(
        "--speaker_json",
        type=str,
        default='./preprocessed_data/VCTK/speakers.json'
    )
    args = parser.parse_args()


    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    import json
    with open(args.speaker_json) as f:
        speaker_map = json.load(f)
    
    # Get dataset
    dataset = TextDataset(args.source, preprocess_config)
    batchs = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=dataset.collate_fn,
    )
    
    control_values = args.duration_control
    pred,gt,out_path,ssl_path,basenames,spks=infer_batch(model,batchs,control_values,args.target_dir)
    

    convert_path='results'
    with open(os.path.join(convert_path,'vctk_result_list.txt'),'w',encoding='utf-8')as fw:
        for i in range(len(ssl_path)):
            print("{}|{}|{}".format(basenames[i],os.path.abspath(out_path[i]),spks[i]),file=fw)
            print("{}_gt|{}|{}".format(basenames[i],os.path.abspath(ssl_path[i]),spks[i]),file=fw)

