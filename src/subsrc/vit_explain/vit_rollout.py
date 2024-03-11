import types

import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import List


def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            attention = attention.unsqueeze(0)
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            #去除不相关的部分
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            #把最前面的cls位置留出
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            # 将attention矩阵中对角线得分+1，再归一化
            a = (attention_heads_fused + 1.0 * I) / 2
            b = a / a.sum(dim=-1)

            result = torch.matmul(b, result)

    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0, 1:]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask

def rollout_cross(attentions, discard_ratio, head_fusion):
    mask_list = []
    # result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            attention = attention.unsqueeze(0)
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            # 去除不相关的部分
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            # 去除不相关的部分
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            # 把最前面的cls位置留出
            indices = indices[indices != 0]
            flat[0, indices] = 0

            # I = torch.eye(attention_heads_fused.size(-1))
            # # 将attention矩阵中对角线得分+1，再归一化
            # a = (attention_heads_fused + 1.0 * I) / 2
            # b = a / a.sum(dim=-1)
            I = torch.eye(attentions[0].size(-1))
            result = torch.matmul(attention_heads_fused, I)
            mask_list.append(result / result.norm(p=1, dim=-1, keepdim=True))


    # Look at the total attention between the class token,
    # and the image patches
    # mask = result[0, 0, 1:]
    mask_list = torch.cat(mask_list, dim=0)

    # mask = mask_list[-1, 0, 1:]
    mask = mask_list[:, 0, 1:]

    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(-1, width, width).numpy()
    mask = mask / np.max(mask)
    return mask
class PredictBaseDataset(Dataset):
    def __init__(self, text, image, transform, tokenizer):
        super(PredictBaseDataset, self).__init__()
        self.image_tensor = transform(image).unsqueeze(0)
        self.encodings = tokenizer(
            text,
            padding='max_length',
            add_special_tokens=True,
            max_length=40,
            truncation=True,
            return_special_tokens_mask=True,  # 遮住特殊token的mask
            return_tensors='pt',
        )

    def __getitem__(self, index):
        # 在训练时，只需返回一张图片和一段文本的图文对
        # 测试时，返回一张图片对应的一组文本
        ret = dict()
        ret['image'] = self.image_tensor
        ret['text_encodings'] = self.encodings
        return ret

    def collate(self, batch):
        return batch[0]

    def __len__(self):
        return 1

class VITAttentionRollout():
    def __init__(self, trainer, model, attention_layer_name='dropout', head_fusion="mean",
                 discard_ratio=0.9):
        self.trainer = trainer
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

        attn_names = ['i2i_p1', 'i2i_p2', 't2t_p1', 't2t_p2', 'i2t', 't2i']
        self.create_attention_funtion(attn_names)
        # self.i2i_p1_attentions = []
        # self.t2t_p1_attentions = []
        # self.i2i_p2_attentions = []
        # self.t2t_p2_attentions = []
        # self.i2t_attentions = []
        # self.t2i_attentions = []

        for name, module in self.model.named_modules():
            if attention_layer_name in name and 'aformer' in name and ('attn' in name or '.self.' in name):
                if 'text' in name:
                    if 'cross' in name:
                        module.register_forward_hook(self.get_t2i_attentions)
                    else:
                        module.register_forward_hook(self.get_t2t_attentions)
                else:
                    if 'cross' in name:
                        module.register_forward_hook(self.get_i2t_attentions)
                    else:
                        module.register_forward_hook(self.get_i2i_attentions)


    def get_i2i_attentions(self, module, fea_in, fea_out):
        self.i2i_stage_flag += 1
        if self.i2i_stage_flag % 2 == 1:
            self.i2i_p1_attentions.append(fea_out[0].cpu())
        else:
            self.i2i_p2_attentions.append(fea_out[0].cpu())

    def get_t2t_attentions(self, module, fea_in, fea_out):
        self.t2t_stage_flag += 1
        if self.t2t_stage_flag % 2 == 1:
            self.t2t_p1_attentions.append(fea_out[0].cpu())
        else:
            self.t2t_p2_attentions.append(fea_out[0].cpu())

    def get_i2t_attentions(self, module, fea_in, fea_out):
        if fea_out.size()[-1] != fea_out.size()[-2]:
            self.i2t_attentions.append(fea_out[0].cpu())

    def get_t2i_attentions(self, module, fea_in, fea_out):
        if fea_out.size()[-1] != fea_out.size()[-2]:
            self.t2i_attentions.append(fea_out[0].cpu())

    def __call__(self, data_loader, mode='t2i'):
        with torch.no_grad():
            # output = self.model.predict_step(input_tensor, batch_idx=None)
            output = self.trainer.predict(self.model, data_loader)
            attention = eval(f'self.{mode}_attentions')
        if mode == 't2i' or mode == 'i2t':
            mask = rollout_cross(attention, self.discard_ratio, self.head_fusion)
        else:
            mask = rollout(attention, self.discard_ratio, self.head_fusion)
        return mask

    def create_attention_funtion(self, attn_names: List):
        self.i2i_stage_flag = 0
        self.t2t_stage_flag = 0
        for attn_name in attn_names:
            exec(f'self.{attn_name}_attentions = []')



