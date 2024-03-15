
import os
import torch

from model import Phoneme2SSL, ScheduledOptim

def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = Phoneme2SSL(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["log_seed"],

            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        # print(ckpt["model"].keys())
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param