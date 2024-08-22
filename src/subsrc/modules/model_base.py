
import torch
import pytorch_lightning as pl
import os
from collections import OrderedDict
import re
from pathlib import Path
from typing import Any, Optional, List, Dict
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,)



# import sys
# sys.path.append('..')
from .my_metrics import Accuracy, Scalar
from .dist_utils import concat_all_gather, all_gather_with_grad

class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def on_fit_start(self) -> None:
        print('============================ FIT LOOP ===================================')
        self.training_step_outputs = []

    def on_validation_start(self) -> None:
        print('============================ VALIDATION LOOP ===================================')
        self.validation_step_outputs = []

    def on_test_start(self) -> None:
        print('============================ TEST LOOP ===================================')
        self.test_step_outputs = []

    def create_visual_encoder(self, config):
        raise NotImplementedError("create custom visual encoder")

    def create_text_encoder(self, config):
        raise NotImplementedError("create custom text encoder")


    @classmethod
    def from_pretrained(cls, config):
        raise NotImplementedError("load from pretrained model")

    @classmethod
    def from_checkpoint(cls, config):
        raise NotImplementedError("load from personal checkpoint")

    def get_state_dict(self, checkpoint_path, use_ema=False):
        if checkpoint_path and os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict_key = 'state_dict'
            if isinstance(checkpoint, dict):
                if use_ema and 'state_dict_ema' in checkpoint:
                    state_dict_key = 'state_dict_ema'
            if state_dict_key and state_dict_key in checkpoint:
                new_state_dict = OrderedDict()
                for k, v in checkpoint[state_dict_key].items():
                    # strip `module.` prefix
                    name = k[7:] if k.startswith('module') else k
                    new_state_dict[name] = v
                state_dict = new_state_dict
            else:
                state_dict = checkpoint
            print("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
            return state_dict
        else:
            print("No checkpoint found at '{}'".format(checkpoint_path))
            raise FileNotFoundError()

    def freeze_module(self, module):
        """
        Freezes module's parameters.
        """
        for parameter in module.parameters():
            parameter.requires_grad = False

    def unfreeze_module(self, module):
        """
        Unfreezes module's parameters.
        """
        for parameter in module.parameters():
            parameter.requires_grad = True

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idxs):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat, self.trainer.world_size)
        text_feats = concat_all_gather(text_feat, self.trainer.world_size)

        batch_size = image_feats.shape[0]

        ptr = int(self.ptr_queue)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.ptr_queue[0] = ptr

    def cal_steps(self):
        if self.trainer.max_steps == None or self.trainer.max_epochs != None:
            max_steps = (len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
                         // self.hparams.config['gradient_accumulation_steps'])
        else:
            max_steps = self.trainer.max_steps

        warmup_steps = max(0, self.hparams.config['warmup_steps'])
        if isinstance(warmup_steps, float):
            warmup_steps = int(warmup_steps * max_steps)
        print(f'====== Max steps: {max_steps},\t Warm up steps: {warmup_steps} =========')
        return max_steps, warmup_steps

    def get_scheduler(self, optimizer, warmup_steps, max_steps):
        if self.hparams.config['scheduler'] == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
            )
        elif self.hparams.config['scheduler'] == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
                num_cycles=self.hparams.config['num_cycles']
            )
        elif self.hparams.config['scheduler'] == 'cosine_hard':
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
                num_cycles=self.hparams.config['num_cycles']
            )
        else:
            scheduler = None
        sched = {
            'scheduler': scheduler, 'interval': 'step'
        }
        return sched

    def set_metrics(self):
        for split in ['train', 'val', 'test']:
            for k, v in self.hparams.config['task_name'].items():
                if v < 1:
                    continue
                if k == 'irtr':
                    setattr(self, f"{split}_{k}_loss", Scalar())
                elif k == 'itm':
                    setattr(self, f"{split}_{k}_accuracy", Accuracy())
                    setattr(self, f"{split}_{k}_loss", Scalar())

    def set_tasks(self):
        self.current_tasks = [k for k, v in self.hparams.config['task_name'].items() if v >= 1]

    def epoch_wrapup(self, step_outputs, phase):
        raise NotImplementedError("epoch wrapup for different tasks")



