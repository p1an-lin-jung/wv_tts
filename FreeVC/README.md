# FreeVC: Towards High-Quality Text-Free One-Shot Voice Conversion

 
## Pre-requisites


1. CD into this repo: `cd FreeVC`

2. Install python requirements: `pip install -r requirements.txt`

3. Download [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-large-960h) and modify the paths in `utils.py` to your own paths.   

4. Download the [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) dataset (for training only).

5. Download [HiFi-GAN model](https://github.com/jik876/hifi-gan) and put it under directory 'hifigan/' (for training with SR only)

## Inference Example

Download the pretrained checkpoints and run:

```python
# inference with FreeVC
CUDA_VISIBLE_DEVICES=0 python convert.py --hpfile logs/freevc-nnemb.json --ptfile checkpoints/freevc.pth --txtpath convert.txt --outdir outputs/freevc-nnemb_vctk

```

Each line in `convert.txt` follows the format below.
```
basename|.npy file path|speaker
```


## Training Example

1. Preprocess

```python
python downsample.py --in_dir </path/to/VCTK/wavs>
ln -s dataset/vctk-16k DUMMY

# run this if you want a different train-val-test split
python preprocess_flist.py

# run this if you want to use pretrained speaker encoder
CUDA_VISIBLE_DEVICES=0 python preprocess_spk.py

# run this if you want to train without SR-based augmentation
CUDA_VISIBLE_DEVICES=3 python preprocess_ssl.py

# run these if you want to train with SR-based augmentation
CUDA_VISIBLE_DEVICES=1 python preprocess_sr.py --min 68 --max 72
CUDA_VISIBLE_DEVICES=1 python preprocess_sr.py --min 73 --max 76
CUDA_VISIBLE_DEVICES=2 python preprocess_sr.py --min 77 --max 80
CUDA_VISIBLE_DEVICES=2 python preprocess_sr.py --min 81 --max 84
CUDA_VISIBLE_DEVICES=3 python preprocess_sr.py --min 85 --max 88
CUDA_VISIBLE_DEVICES=3 python preprocess_sr.py --min 89 --max 92
 --min 89 --max 92

```

2. Train

```python
CUDA_VISIBLE_DEVICES=3 python train.py -c configs/freevc-nnemb.json -m freevc-nnemb_vctk
```

## References

- https://github.com/jaywalnut310/vits
- https://github.com/jik876/hifi-gan
- https://github.com/liusongxiang/ppg-vc
