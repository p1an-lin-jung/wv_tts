CUDA_VISIBLE_DEVICES=1 python3 synth_batch.py --restore_step 100  --source ./test_list/p226.txt -p config/VCTK/preprocess.yaml -m config/VCTK/model.yaml -t config/VCTK/train.yaml
