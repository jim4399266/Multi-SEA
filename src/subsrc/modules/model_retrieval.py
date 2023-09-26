import torch.distributed as dist
from torch import nn
import torch
import torch.nn.functional as F
import re
import pytorch_lightning as pl
from typing import Any, Optional, List, Dict
import gc
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertConfig
from pathlib import Path

# from .bert_model import BertCrossLayer, BertAttention
from .clip_model import build_model, adapt_position_encoding
from .dist_utils import concat_all_gather, all_gather_with_grad
# from .blip import create_vit, init_tokenizer, load_checkpoint
from . import train, evaluate
# from .med import BertConfig, BertModel
from .model_base import BaseModule
from .AFormer import AFormer
# from .AFormer_b import AFormer_b

class RetrievalModule(BaseModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        image_encoder_config = config['image_encoder_config']
        text_encoder_config = config['text_encoder_config']
        hidden_size = config['hidden_size']

        self.distill = True
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.negative_all_rank = config['negative_all_rank']

        self.visual_encoder, vision_width = self.create_visual_encoder(image_encoder_config)
        self.text_encoder, text_width = self.create_text_encoder(text_encoder_config)
        self.vision_proj = nn.Linear(vision_width, hidden_size)
        self.text_proj = nn.Linear(text_width, hidden_size)

        aformer_config = BertConfig.from_json_file(config['aformer_config_path'])
        aformer_config.num_hidden_layers = config['num_top_layer']
        self.aformer = AFormer(aformer_config)

        self.itm_head = nn.Linear(hidden_size, 2)

        self.set_metrics()

    def encoding_text(self,batch):
        # 获得预训练模型输出的特征
        text_encoding = batch['text_encodings']
        input_ids = text_encoding['input_ids'].to(self.device)
        attention_mask = text_encoding['attention_mask'].to(self.device)

        text_frozen_embeds = self.text_encoder(input_ids,
                                               attention_mask=attention_mask,
                                               return_dict=True)
        text_frozen_embeds = F.normalize(self.text_proj(text_frozen_embeds.last_hidden_state), dim=-1)

        # 通过 AFormer 输出特征向量
        text_outputs = self.aformer(
            text_frozen_embeds,
            attention_mask=attention_mask,
            mode='text'
        )
        text_feats = text_outputs.pooler_output
        text_embeds = text_outputs.last_hidden_state
        return [text_feats, text_embeds, attention_mask]

    def encoding_image(self, batch):
        # 获得预训练模型输出的特征
        image = batch['image'].to(self.device)

        image_frozen_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_frozen_embeds.size()[:-1], dtype=torch.long).to(self.device)
        image_frozen_embeds = F.normalize(self.vision_proj(image_frozen_embeds), dim=-1)

        # 通过 AFormer 输出特征向量
        image_outputs = self.aformer(
            image_frozen_embeds,
            mode='image'
        )
        image_feats = image_outputs.pooler_output
        image_embeds = image_outputs.last_hidden_state
        return [image_feats, image_embeds, image_atts]

    def forward(self, batch, phase):
        return train.train_irtr(self, batch)

    def training_step(self, batch, batch_idx):
        if self.trainer.current_epoch > 0:
            alpha = self.hparams.config['alpha']
        else:
            alpha = self.hparams.config['alpha'] * min(1, batch_idx / len(self.trainer.datamodule.train_dataloader()))
        self.hparams.config['cur_alpha'] = alpha

        irtr_loss = self(batch, phase='train')

        lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0 \
                and batch_idx % self.trainer.accumulate_grad_batches == 0:
            self.print('Global step:{global_step}.'
                       'Train Loss: {loss:.4f} '
                       'LR: {lr:.3E}'
                       .format(global_step=self.trainer.global_step,
                               loss=irtr_loss,
                               lr=lr))
        return irtr_loss

    def on_train_epoch_end(self) -> None:
        self.epoch_wrapup(phase='train')
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self) -> None:
        # 不传入out了，直接从self.validation_step_outputs获取每个val step的返回
        # all_preds = torch.stack(self.validation_step_outputs)
        self.epoch_wrapup(phase='val')
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        pass

    def on_test_epoch_end(self) -> None:
        self.epoch_wrapup(phase='test')

    def epoch_wrapup(self, phase):
        the_metric = 0
        total_loss = 0
        if not self.training:
            if phase == 'val':
                data_loader = self.trainer.datamodule.val_dataloader()
                cur_step = self.global_step
            else:
                data_loader = self.trainer.datamodule.test_dataloader()
                patt = re.compile("step(\d*)")
                cur_step = re.search(patt, Path(self.trainer.ckpt_path).stem).group(1)
            score_val_i2t, score_val_t2i = evaluate.val_irtr(self, data_loader)
            val_result = evaluate.recall_eval(score_val_i2t, score_val_t2i, data_loader.dataset.index_mapper)
            print(f'global_step: {cur_step}')
            print(val_result)
            for item in ['txt_r1', 'txt_r5', 'txt_r10', 'txt_r_mean', 'img_r1', 'img_r5', 'img_r10', 'img_r_mean', 'r_mean']:
                self.logger.experiment.add_scalar(f"{phase}/{item}", val_result[item], cur_step)

            # the_metric += (val_result['txt_r1'] + val_result['img_r1']) * 10 \
            #               + (val_result['txt_r5'] + val_result['img_r5']) * 5 \
            #               + val_result['txt_r10'] + val_result['img_r10']

            the_metric += (val_result['r_sum'])

            self.logger.experiment.add_scalar(
                f'{phase}/the_metric', the_metric, cur_step
            )
            # self.log(f'{phase}/irtr/the_metric', the_metric)
            # self.log(f'{phase}/r_mean', val_result['r_mean'])
            self.log(f'{phase}/the_metric', the_metric)

    def create_text_encoder(self, config):
        '''仅创建模型，不加载权重'''
        backbone_config = AutoConfig.from_pretrained(config['tokenizer'], output_hidden_states=True)
        backbone = AutoModel.from_config(backbone_config)
        return backbone, backbone_config.hidden_size

    def create_visual_encoder(self, config):
        is_clip = (not 'swin' in config['vit'])
        # 两个模态的编码器，使用预训练好的模型
        if is_clip:
            backbone = build_model(config['vit'], resolution_after=config['image_size'])
            vision_width = backbone.visual.width
        else:
            backbone, vision_width = None, None
        return backbone, vision_width

    def configure_optimizers(self):
        opt_config = self.hparams.config['optimizer']
        max_steps, warmup_steps = self.cal_steps()
        optimizer = torch.optim.AdamW(params=self.parameters(),
                                      lr=opt_config['init_lr'],
                                      weight_decay=opt_config['weight_decay'],
                                      eps=opt_config['eps'],
                                      betas=opt_config['betas'])
        sched = self.get_scheduler(optimizer, warmup_steps, max_steps)
        return {
            'optimizer': optimizer,
            'lr_scheduler': sched,
        }

    @classmethod
    def from_pretrained(cls, config):
        model = cls(config)
        model.text_encoder = AutoModel.from_pretrained(config['text_encoder_config']['tokenizer'])
        print('### load model from pretrained! ###')
        model.freeze_module(model.text_encoder)
        model.freeze_module(model.visual_encoder)
        print('### freeze text encoder and visual encoder.')
        return model

    @classmethod
    def from_checkpoint(cls, config):
        model = cls(config)
        # state_dict = load_checkpoint(model, config['pretrained'])
        # msg = model.load_state_dict(state_dict, strict=False)
        # print("missing keys:")
        # print(msg.missing_keys)
        # model.copy_params()
        # model.set_queue()
        return model

class RetrievalModuleWithQueue(BaseModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        image_encoder_config = config['image_encoder_config']
        text_encoder_config = config['text_encoder_config']
        hidden_size = config['hidden_size']

        self.distill = config['distill']
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.queue_size = config['queue_size']
        self.negative_all_rank = config['negative_all_rank']

        self.visual_encoder, vision_width = self.create_visual_encoder(image_encoder_config)
        self.text_encoder, text_width = self.create_text_encoder(text_encoder_config)
        self.vision_proj = nn.Linear(vision_width, hidden_size)
        self.text_proj = nn.Linear(text_width, hidden_size)

        aformer_config = BertConfig.from_json_file(config['aformer_config_path'])
        aformer_config.num_hidden_layers = config['num_top_layer']
        self.aformer = AFormer(aformer_config)

        self.itm_head = nn.Linear(hidden_size, 2)

        self.set_metrics()\

        #create the queue
        self.register_buffer("image_queue", torch.rand(hidden_size, self.queue_size))
        self.register_buffer("text_queue", torch.rand(hidden_size, self.queue_size))
        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def encoding_text(self,batch):
        # 获得预训练模型输出的特征
        text_encoding = batch['text_encodings']
        input_ids = text_encoding['input_ids'].to(self.device)
        attention_mask = text_encoding['attention_mask'].to(self.device)

        text_frozen_embeds = self.text_encoder(input_ids,
                                               attention_mask=attention_mask,
                                               return_dict=True)
        text_frozen_embeds = F.normalize(self.text_proj(text_frozen_embeds.last_hidden_state), dim=-1)

        # 通过 AFormer 输出特征向量
        text_outputs = self.aformer(
            text_frozen_embeds,
            attention_mask=attention_mask,
            mode='text'
        )
        text_feats = text_outputs.pooler_output
        text_embeds = text_outputs.last_hidden_state
        return [text_feats, text_embeds, attention_mask]

    def encoding_image(self, batch):
        # 获得预训练模型输出的特征
        image = batch['image'].to(self.device)

        image_frozen_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_frozen_embeds.size()[:-1], dtype=torch.long).to(self.device)
        image_frozen_embeds = F.normalize(self.vision_proj(image_frozen_embeds), dim=-1)

        # 通过 AFormer 输出特征向量
        image_outputs = self.aformer(
            image_frozen_embeds,
            mode='image'
        )
        image_feats = image_outputs.pooler_output
        image_embeds = image_outputs.last_hidden_state
        return [image_feats, image_embeds, image_atts]

    def forward(self, batch, phase):
        return train.train_irtr_with_queue(self, batch)

    def training_step(self, batch, batch_idx):
        if self.trainer.current_epoch > 0:
            alpha = self.hparams.config['alpha']
        else:
            alpha = self.hparams.config['alpha'] * min(1, batch_idx / len(self.trainer.datamodule.train_dataloader()))
        self.hparams.config['cur_alpha'] = alpha

        irtr_loss = self(batch, phase='train')

        lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0 \
                and batch_idx % self.trainer.accumulate_grad_batches == 0:
            self.print('Global step:{global_step}.'
                       'Train Loss: {loss:.4f} '
                       'LR: {lr:.3E}'
                       .format(global_step=self.trainer.global_step,
                               loss=irtr_loss,
                               lr=lr))
        return irtr_loss

    def on_train_epoch_end(self) -> None:
        self.epoch_wrapup(phase='train')
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self) -> None:
        # 不传入out了，直接从self.validation_step_outputs获取每个val step的返回
        # all_preds = torch.stack(self.validation_step_outputs)
        self.epoch_wrapup(phase='val')
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        pass

    def on_test_epoch_end(self) -> None:
        self.epoch_wrapup(phase='test')

    def epoch_wrapup(self, phase):
        the_metric = 0
        total_loss = 0
        if not self.training:
            if phase == 'val':
                data_loader = self.trainer.datamodule.val_dataloader()
                cur_step = self.global_step
            else:
                data_loader = self.trainer.datamodule.test_dataloader()
                patt = re.compile("step(\d*)")
                cur_step = re.search(patt, Path(self.trainer.ckpt_path).stem).group(1)
            val_result = evaluate.val_irtr(self, data_loader)
            print(f'global_step: {cur_step}')
            print(val_result)

            for item in ['txt_r1', 'txt_r5', 'txt_r10', 'txt_r_mean', 'img_r1', 'img_r5', 'img_r10', 'img_r_mean', 'r_mean']:
                self.logger.experiment.add_scalar(f"{phase}_{dataset}{sacle}/{item}", val_result[item], cur_step)

            the_metric += (val_result['r_sum'])

            self.logger.experiment.add_scalar(
                f'{phase}/the_metric', the_metric, cur_step
            )
            # self.log(f'{phase}/irtr/the_metric', the_metric)
            # self.log(f'{phase}/r_mean', val_result['r_mean'])
            self.log(f'{phase}/the_metric', the_metric)

    def create_text_encoder(self, config):
        '''仅创建模型，不加载权重'''
        backbone_config = AutoConfig.from_pretrained(config['tokenizer'], output_hidden_states=True)
        backbone = AutoModel.from_config(backbone_config)
        return backbone, backbone_config.hidden_size

    def create_visual_encoder(self, config):
        is_clip = (not 'swin' in config['vit'])
        # 两个模态的编码器，使用预训练好的模型
        if is_clip:
            backbone = build_model(config['vit'], resolution_after=config['image_size'])
            vision_width = backbone.visual.width
        else:
            backbone, vision_width = None, None
        return backbone, vision_width

    def configure_optimizers(self):
        opt_config = self.hparams.config['optimizer']
        max_steps, warmup_steps = self.cal_steps()
        optimizer = torch.optim.AdamW(params=self.parameters(),
                                      lr=opt_config['init_lr'],
                                      weight_decay=opt_config['weight_decay'],
                                      eps=opt_config['eps'],
                                      betas=opt_config['betas'])
        sched = self.get_scheduler(optimizer, warmup_steps, max_steps)
        return {
            'optimizer': optimizer,
            'lr_scheduler': sched,
        }

    @classmethod
    def from_pretrained(cls, config):
        model = cls(config)
        model.text_encoder = AutoModel.from_pretrained(config['text_encoder_config']['tokenizer'])
        print('### load model from pretrained! ###')
        model.freeze_module(model.text_encoder)
        model.freeze_module(model.visual_encoder)
        print('### freeze text encoder and visual encoder.')
        return model

    @classmethod
    def from_checkpoint(cls, config):
        model = cls(config)
        # state_dict = load_checkpoint(model, config['pretrained'])
        # msg = model.load_state_dict(state_dict, strict=False)
        # print("missing keys:")
        # print(msg.missing_keys)
        # model.copy_params()
        # model.set_queue()
        return model

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idx):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat, self.trainer.world_size)
        text_feats = concat_all_gather(text_feat, self.trainer.world_size)
        idxs = concat_all_gather(idx, self.trainer.world_size)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        # assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        step = batch_size if (ptr + batch_size) <= self.queue_size else (
                    batch_size - (ptr + batch_size - self.queue_size))
        self.image_queue[:, ptr:ptr + step] = image_feats[:step].T
        self.text_queue[:, ptr:ptr + step] = text_feats[:step].T
        self.idx_queue[:, ptr:ptr + step] = idxs[:step].T
        ptr = (ptr + step) % self.queue_size  # move pointer

        # try:
        #     # step = batch_size if (ptr + batch_size) <= self.queue_size else (batch_size - (ptr + batch_size - self.queue_size))
        #     self.idx_queue[:, ptr:ptr + step] = idxs[:step].T
        #     ptr = (ptr + step) % self.queue_size  # move pointer
        # except:
        #     print('----------------------------------------------')
        #     print(f'queue_size:{self.queue_size}, ptr:{ptr}, step:{step}')
        #     print(idxs[:step])
        #     print('---------')
        #     print(idxs[:step].T)

        self.queue_ptr[0] = ptr

