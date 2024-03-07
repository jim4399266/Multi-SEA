import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset


def rollout(attentions, discard_ratio, head_fusion):

    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            attention = attention.mean(axis=0)
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
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0, 1:]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).numpy()
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


class VITAttentionRollout:
    def __init__(self, trainer, model, attention_layer_name='dropout', head_fusion="mean",
                 discard_ratio=0.9):
        self.trainer = trainer
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

        self.i2i_attentions = []
        self.t2t_attentions = []
        self.i2t_attentions = []
        self.t2i_attentions = []

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

    def get_t2i_attentions(self, module, fea_in, fea_out):
        self.t2i_attentions.append(fea_out[0][1:].cpu())

    def get_t2t_attentions(self, module, fea_in, fea_out):
        self.t2t_attentions.append(fea_out[0][1:].cpu())

    def get_i2t_attentions(self, module, fea_in, fea_out):
        self.i2t_attentions.append(fea_out[0][1:,1:].cpu())

    def get_i2i_attentions(self, module, fea_in, fea_out):
        self.i2i_attentions.append(fea_out[0][1:,1:].cpu())


    def __call__(self, data_loader):
        with torch.no_grad():
            # output = self.model.predict_step(input_tensor, batch_idx=None)
            output = self.trainer.predict(self.model, data_loader)


        return rollout(self.i2i_attentions, self.discard_ratio, self.head_fusion)