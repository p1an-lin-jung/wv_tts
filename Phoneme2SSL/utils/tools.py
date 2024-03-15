import torch
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(data, device):
    if len(data) == 10:
        (
            ids,
            raw_texts,
            ssl_embs,
            speakers,
            texts,
            src_lens,
            max_src_len,
            ssl_lens,
            max_ssl_len,
            ssl_durations,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        ssl_embs=torch.from_numpy(ssl_embs).to(device)
        ssl_lens=torch.from_numpy(ssl_lens).to(device)
        ssl_durations=torch.from_numpy(ssl_durations).to(device)
        
        return (
            ids,
            raw_texts,
            ssl_embs,
            speakers,
            texts,
            src_lens,
            max_src_len,
            ssl_lens,
            max_ssl_len,
            ssl_durations,
        )

    if len(data) == 8:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len,ssl_lens,ssl_path) = data
        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        ssl_lens = torch.LongTensor(ssl_lens).to(device)
        
        return (ids, raw_texts, speakers, texts, src_lens, max_src_len,ssl_lens,ssl_path)

def log(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag="",
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/ssl_loss", losses[1], step)
        logger.add_scalar("Loss/ssl_postnet_loss", losses[2], step)
        logger.add_scalar("Loss/duration_loss", losses[3], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask

def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)

def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
