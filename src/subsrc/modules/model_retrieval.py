import torch.distributed as dist
import copy
from torch import nn
import torch
import torch.nn.functional as F
import re
import pytorch_lightning as pl
from typing import Any, Optional, List, Dict
import gc
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertConfig
from transformers import ViTImageProcessor, ViTModel
from pathlib import Path
from collections import defaultdict
# from torchinfo import summary
from copy import deepcopy

from .clip_model import build_model, adapt_position_encoding
from .dist_utils import concat_all_gather, all_gather_with_grad
from . import train, evaluate
from .model_base import BaseModule
from .AFormer import AFormerWithAug, Pooler, Swish


class RetrievalModuleWithQueue_1(BaseModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        image_encoder_config = config['image_encoder_config']
        text_encoder_config = config['text_encoder_config']
        hidden_size = config['hidden_size']

        self.distill = config['distill']
        self.temp = nn.Parameter(1.0 * torch.ones([]))
        self.alpha = nn.Parameter(0.7 * torch.ones([]))
        self.queue_size = config['queue_size']
        self.negative_all_rank = config['negative_all_rank']

        self.visual_encoder, vision_width = self.create_visual_encoder(image_encoder_config)
        self.text_encoder, text_width = self.create_text_encoder(text_encoder_config)
        self.vision_proj_f = nn.Linear(vision_width, hidden_size)
        self.vision_proj_e = nn.Linear(vision_width, hidden_size)
        self.vision_pooler = Pooler(vision_width)
        self.text_proj_f = nn.Linear(text_width, hidden_size)
        self.text_proj_e = nn.Linear(text_width, hidden_size)
        self.swish = Swish(config['beta'])

        aformer_config = BertConfig.from_json_file(config['aformer_config_path'])
        aformer_config.num_hidden_layers = config['num_top_layer']
        aformer_config.num_attention_heads = config['num_heads']
        aformer_config.hidden_size = config['hidden_size']
        aformer_config.encoder_width = config['hidden_size']
        aformer_config.intermediate_size = config['hidden_size'] * config['mlp_ratio']
        aformer_config.attention_groups = config['attention_groups']
        aformer_config.beta = config['beta']
        aformer_config.attention_probs_dropout_prob = config['drop_rate']

        self.aformer = AFormerWithAug(aformer_config)

        self.itm_head = nn.Linear(hidden_size, 2)

        self.set_metrics()

        #create the queue
        self.register_buffer("image_queue", torch.rand(hidden_size, self.queue_size))
        self.register_buffer("text_queue", torch.rand(hidden_size, self.queue_size))
        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def encoding_text(self,batch):
        text_encoding = batch['text_encodings']
        input_ids = text_encoding['input_ids'].to(self.device)
        attention_mask = text_encoding['attention_mask'].to(self.device)

        pretrained_output = self.text_encoder(input_ids,
                                               attention_mask=attention_mask,
                                               return_dict=True)

        pretrained_feats = self.text_proj_f(pretrained_output.pooler_output)
        pretrained_embeds = self.text_proj_e(pretrained_output.last_hidden_state)


        text_outputs = self.aformer(
            pretrained_embeds,
            attention_mask=attention_mask,
            mode='text'
        )
        text_feats = text_outputs.pooler_output * self.alpha + pretrained_feats * (1 - self.alpha)
        text_embeds = text_outputs.last_hidden_state * self.alpha + pretrained_embeds * (1 - self.alpha)

        return [text_feats, text_embeds, attention_mask]

    def encoding_image(self, batch):

        image = batch['image'].to(self.device)

        pretrained_output = self.visual_encoder(image)
        if hasattr(pretrained_output, 'last_hidden_state'):
            pretrained_feats = self.vision_proj_f(pretrained_output.pooler_output)
            pretrained_embeds = self.vision_proj_e(pretrained_output.last_hidden_state)
        else:
            pretrained_feats = self.vision_proj_f(self.vision_pooler(pretrained_output))
            pretrained_embeds = self.vision_proj_e(pretrained_output)

        image_atts = torch.ones(pretrained_embeds.size()[:-1], dtype=torch.long).to(self.device)

        image_outputs = self.aformer(
            pretrained_embeds,
            mode='image'
        )

        image_feats = image_outputs.pooler_output * self.alpha + pretrained_feats * (1 - self.alpha)
        image_embeds = image_outputs.last_hidden_state * self.alpha + pretrained_embeds * (1 - self.alpha)
        return [image_feats, image_embeds, image_atts]

    def forward(self, batch, phase):
        return train.train_irtr_with_queue_multi_out(self, batch)

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
        self.epoch_wrapup(None, phase='train')
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        # pass
        image_feats, image_embeds, image_atts = self.encoding_image(batch)
        text_feats, text_embeds, text_atts = self.encoding_text(batch)
        self.validation_step_outputs.append([text_embeds, text_feats, text_atts,
                                             image_embeds, image_feats, image_atts])


    def on_validation_epoch_end(self) -> None:
        self.epoch_wrapup(self.validation_step_outputs, phase='val')
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        # pass
        image_feats, image_embeds, image_atts = self.encoding_image(batch)
        text_feats, text_embeds, text_atts = self.encoding_text(batch)
        self.test_step_outputs.append([text_embeds, text_feats, text_atts,
                                             image_embeds, image_feats, image_atts])

    def on_test_epoch_end(self) -> None:
        self.epoch_wrapup(self.test_step_outputs, phase='test')
        self.test_step_outputs.clear()  # free memory

    def epoch_wrapup(self, step_outputs, phase):
        the_metric = 0
        dataset = self.hparams.config['datasets'][0]
        if not self.training:
            if phase == 'val':
                cur_step = self.global_step

                index_mapper = self.trainer.datamodule.val_dataloader().dataset.index_mapper
                image_mapper = self.trainer.datamodule.val_dataloader().dataset.image_mapper
            else:
                patt = re.compile("step(\d*)")
                cur_step = 0 if self.trainer.ckpt_path == None else \
                    re.search(patt, Path(self.trainer.ckpt_path).stem).group(1)
                index_mapper = self.trainer.datamodule.test_dataloader().dataset.index_mapper
                image_mapper = self.trainer.datamodule.test_dataloader().dataset.image_mapper

            vectors = list(zip(*step_outputs))
            vectors = [torch.cat(vec, dim=0) for vec in vectors]
            step_outputs.clear()
            torch.cuda.empty_cache()


            if '1k' in self.hparams.config['coco_scale']:
                val_result = defaultdict(list)
                for i in range(5):
                    assert len(vectors[-1]) % 5 == 0, "图片无法均匀切分"

                    image_step = int(len(vectors[-1]) / 5)
                    text_s, text_e = image_mapper[i * image_step][0], image_mapper[(i + 1) * image_step - 1][-1] + 1

                    sub_index_mapper = dict()
                    for j in range(text_s, text_e):
                        sub_index_mapper[j - text_s] = copy.deepcopy(index_mapper[j])
                        sub_index_mapper[j - text_s][0] -= i * image_step

                    sub_text_vectors = [embeds[text_s:text_e] for embeds in vectors[:3]]
                    sub_image_vectors = [embeds[i * image_step:(i + 1) * image_step] for embeds in vectors[3:]]
                    sub_vectors = sub_text_vectors + sub_image_vectors

                    score_val_i2t, score_val_t2i = evaluate.val_irtr_recall_sort(self, sub_vectors)
                    sub_val_result = evaluate.calculate_score(score_val_i2t, score_val_t2i,
                                                          sub_index_mapper)
                    for k, v in sub_val_result.items():
                        val_result[k].append(v)
                for k, v in val_result.items():
                    val_result[k] = np.mean(v)
                    self.logger.experiment.add_scalar(f"{phase}_{dataset}1k/{k}", np.mean(v), cur_step)
                print(val_result)

            if '5k' in self.hparams.config['coco_scale']:
                score_val_i2t, score_val_t2i = evaluate.val_irtr_recall_sort(self, vectors)
                val_result = evaluate.calculate_score(score_val_i2t, score_val_t2i, index_mapper)
                for k, v in val_result.items():
                    self.logger.experiment.add_scalar(f"{phase}_{dataset}5k/{k}", np.mean(v), cur_step)
                print(val_result)
            if self.hparams.config['coco_scale'] == ['']:
                score_val_i2t, score_val_t2i = evaluate.val_irtr_recall_sort(self, vectors)
                val_result = evaluate.calculate_score(score_val_i2t, score_val_t2i, index_mapper)
                for k, v in val_result.items():
                    self.logger.experiment.add_scalar(f"{phase}_{dataset}1k/{k}", np.mean(v), cur_step)
                print(val_result)

            # calculate total scores
            print(f'global_step: {cur_step}')
            the_metric += (val_result['r_sum'])
            self.logger.experiment.add_scalar(
                f'{phase}/the_metric', the_metric, cur_step
            )
            self.log(f'{phase}/the_metric', the_metric)


    def create_text_encoder(self, config):
        '''create model without weights'''
        backbone_config = AutoConfig.from_pretrained(config['tokenizer'], output_hidden_states=True)
        backbone = AutoModel.from_config(backbone_config)
        return backbone, backbone_config.hidden_size

    def create_visual_encoder(self, config):
        is_clip = ('clip' in config['train_transform_keys'])
        if is_clip:
            backbone = build_model(config['vit'], resolution_after=config['image_size'])
            vision_width = backbone.visual.width
        else:
            backbone = ViTModel.from_pretrained(config['vit'])
            vision_width = backbone.config.hidden_size
        return backbone, vision_width

    def configure_optimizers(self):
        opt_config = self.hparams.config['optimizer']
        max_steps, warmup_steps = self.cal_steps()
        optimizer_grouped_parameters = [
            {
                "name": 'default',
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in ["alpha", "temp", "text_encoder", "visual_encoder"])

                ]
            },
            {
                "name": 'encoder',
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in ["text_encoder", "visual_encoder"])
                ],
                "lr": opt_config['init_lr'] / 10,

            },
            {
                "name": 'alpha',
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in ["alpha"])
                ],
                "lr": 1e-4,
            },
            {
                "name": 'temp',
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in ["temp"])
                ],
                "lr": 5e-4,
            },
            ]
        optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters,
                                      lr=opt_config['init_lr'],
                                      weight_decay=opt_config['weight_decay'],
                                      eps=opt_config['eps'],
                                      betas=opt_config['betas'])
        sched = self.get_scheduler(optimizer, warmup_steps, max_steps)
        return {
            'optimizer': optimizer,
            'lr_scheduler': sched,
        }

    def freeze_text_encoder(self, module, last_layer=0):
        self.freeze_module(module)
        if last_layer > 0:
            self.unfreeze_module(module.encoder.layer[-last_layer:])

    def freeze_image_encoder(self, module, last_layer=0):
        self.freeze_module(module)
        if last_layer > 0:
            self.unfreeze_module(module.visual.transformer.resblocks[-last_layer:])
            self.unfreeze_module(module.visual.ln_post)

    @classmethod
    def from_pretrained(cls, config):
        model = cls(config)
        model.text_encoder = AutoModel.from_pretrained(config['text_encoder_config']['tokenizer'])
        print('### load model from pretrained! ###')
        model.freeze_image_encoder(model.visual_encoder, last_layer=4)
        print('### freeze text encoder and visual encoder.')
        return model

    @classmethod
    def from_checkpoint(cls, config, strict=True):
        model = cls(config)
        state_dict = model.get_state_dict(config['checkpoint'])
        msg = model.load_state_dict(state_dict, strict=strict)
        print("missing keys:")
        print(msg.missing_keys)
        return model

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idx):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat, self.trainer.world_size)
        text_feats = concat_all_gather(text_feat, self.trainer.world_size)
        idxs = concat_all_gather(idx, self.trainer.world_size)
        batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        step = batch_size if (ptr + batch_size) <= self.queue_size else (
                    batch_size - (ptr + batch_size - self.queue_size))
        self.image_queue[:, ptr:ptr + step] = image_feats[:step].T
        self.text_queue[:, ptr:ptr + step] = text_feats[:step].T
        self.idx_queue[:, ptr:ptr + step] = idxs[:step].T
        ptr = (ptr + step) % self.queue_size  # move pointer


        self.queue_ptr[0] = ptr

