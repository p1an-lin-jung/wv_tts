import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_param_num
from utils.tools import to_device, log
from model import P2SLoss
from dataset import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset


    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    # print(model)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = P2SLoss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)


    # Init logger
    # for p in train_config["path"].values():
    os.makedirs(os.path.join(train_config["path"]["log_seed"]), exist_ok=True)
    os.makedirs(os.path.join(train_config["path"]["log_seed"],train_config["path"]["ckpt_path"]), exist_ok=True)
    os.makedirs(os.path.join(train_config["path"]["log_seed"],train_config["path"]["log_path"]), exist_ok=True)
    os.makedirs(os.path.join(train_config["path"]["log_seed"],train_config["path"]["result_path"]), exist_ok=True)


    train_log_path = os.path.join(train_config["path"]["log_seed"],train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_seed"],train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    # val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                # from text import _id_to_symbol
                batch = to_device(batch, device)
                
                # Forward
                output = model(*(batch[3:]))

                # Cal Loss
                losses = Loss(batch, output)
                total_loss = losses[0]
                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)

                    message2 = "Total Loss: {:.4f}, ssl Loss: {:.4f}, ssl PostNet Loss: {:.4f},ssl Duration Loss: {:.4f}".format(
                            *losses
                        )

                    dur = torch.exp(output[2])-1

                    with open(os.path.join(train_log_path, "log_train.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")
                        print(dur[0][:20],file=f)
                        print(batch[-1][0][:20],file=f)
                    outer_bar.write(message1 + message2)
                    log(train_logger, step, losses=losses)

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["log_seed"],
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
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
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
