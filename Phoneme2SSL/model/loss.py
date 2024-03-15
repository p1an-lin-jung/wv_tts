import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F

class P2SLoss(nn.Module):

    def __init__(self, preprocess_config, model_config):
        super(P2SLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.model_config=model_config

    def forward(self, inputs, predictions):

        (
            ssl_emb_target,
            _,
            _,
            _,
            _,
            _,
            _,
            ssl_durations_target,
        ) = inputs[2:]

        (
            ssl_predictions,
            postnet_ssl_predictions,
            ssl_log_duration_predictions,
            _,
            src_masks,
            ssl_masks,
            _,
            _,
            _
        ) = predictions


        src_masks = ~src_masks
        ssl_masks = ~ssl_masks
        ssl_log_duration_targets = torch.log(ssl_durations_target.float() + 1)

        ssl_emb_target = ssl_emb_target[:, : ssl_masks.shape[1], :]
        ssl_masks = ssl_masks[:, :ssl_masks.shape[1]]

        ssl_log_duration_targets.requires_grad = False
        ssl_emb_target.requires_grad = False

        ssl_log_duration_targets = ssl_log_duration_targets.masked_select(src_masks)
        ssl_log_duration_predictions =ssl_log_duration_predictions.masked_select(src_masks)

        ssl_predictions = ssl_predictions.masked_select(ssl_masks.unsqueeze(-1))
        postnet_ssl_predictions = postnet_ssl_predictions.masked_select(
            ssl_masks.unsqueeze(-1)
        )
        ssl_emb_target = ssl_emb_target.masked_select(ssl_masks.unsqueeze(-1))


        ssl_duration_loss = self.mse_loss(ssl_log_duration_predictions, ssl_log_duration_targets)

        ssl_loss =self.mae_loss(ssl_predictions, ssl_emb_target)
        postnet_ssl_loss =self.mae_loss(postnet_ssl_predictions, ssl_emb_target)

        total_loss = (
            ssl_loss + postnet_ssl_loss  + ssl_duration_loss 
        )

        return (
            total_loss,
            ssl_loss,
            postnet_ssl_loss,            
            ssl_duration_loss,
        )