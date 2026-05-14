import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pos_utils import FreqCondNetwork
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


class FrequencyExpandModel:
    def __init__(self, freq_dim=50):
        self.deform = FreqCondNetwork(xyz_dim=3, freq_dim=freq_dim, hidden_dim=256, cond_dim=128).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 1

    def step(self, xyz, freq_emb):
        return self.deform(xyz, freq_emb)

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.d_r_init * self.spatial_lr_scale,
             "name": "deform"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.d_r_init * self.spatial_lr_scale,
                                                       lr_final=training_args.d_r_final,
                                                       lr_delay_mult=training_args.d_lr_delay_mult,
                                                       max_steps=training_args.d_lr_max_steps)

    def save_weights(self, model_path, iteration, freq_idx):
        out_weights_path = os.path.join(model_path, "wideband_deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        ckpt = {
            "iteration": iteration,
            "model_state": self.deform.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,
        }
        torch.save(ckpt, os.path.join(out_weights_path, 'wideband_deform.pth'))

    def load_weights(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cuda")
        self.deform.load_state_dict(checkpoint["model_state"])
        if self.optimizer is not None and checkpoint["optimizer_state"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        print(f"[INFO] Loaded weights from {ckpt_path}, iteration {checkpoint['iteration']}")
        return checkpoint["iteration"]


    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr



