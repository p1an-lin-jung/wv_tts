import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor_f1
from utils.tools import get_mask_from_lengths
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Phoneme2SSL(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(Phoneme2SSL, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
       
        self.variance_adaptor = VarianceAdaptor_f1(preprocess_config, model_config)
        self.ssl_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["w2v"]["n_channels"],
            )
        self.postnet = PostNet(n_mel_channels=preprocess_config["preprocessing"]["w2v"]["n_channels"])
        self.decoder = Decoder(model_config)
    
        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        ssl_lens=None,
        max_ssl_len=None,
        ssl_durations=None,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        
        pdb.set_trace()
        ssl_masks= (
            get_mask_from_lengths(ssl_lens, max_ssl_len)
            if ssl_lens is not None
            else None
        )
        output = self.encoder(texts, src_masks) #[b,max_src_len,256(dim)]


        if self.speaker_emb is not None:
            spk_emb=self.speaker_emb(speakers)

            # (batch_size, n_frames, n_channels)             
            spk_emb_expd=spk_emb.unsqueeze(1).expand(
                -1, max_src_len, -1
            )
            output = output + spk_emb_expd
        else:
            spk_emb=None

        (
            output,
            log_d_predictions,
            d_rounded,
            ssl_lens,
            ssl_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            ssl_masks,
            max_ssl_len,
            ssl_durations,
            d_control,
        )
        output, ssl_masks = self.decoder(output, ssl_masks)
        output = self.ssl_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            log_d_predictions,
            d_rounded,
            src_masks,
            ssl_masks,
            src_lens,
            ssl_lens,
            spk_emb
        )
