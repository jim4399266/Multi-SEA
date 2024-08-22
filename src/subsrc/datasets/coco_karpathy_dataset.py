# from .base_dataset import BaseDataset
import io
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
from typing import Union, Optional, List, Dict

# import sys
# sys.path.append('..')
from .base_dataset import CocoKarpathyBaseDataset


class CocoKarpathyDataset(CocoKarpathyBaseDataset):
    def __init__(self, *args, split='', names='', **kwargs):
        assert split in ['train', 'val', 'test']
        self.split = split
        if split == "train":
            names = ["coco_caption_karpathy_train", "coco_caption_karpathy_restval"]
        elif split == "val":
            names = ["coco_caption_karpathy_val"]
        elif split == "test":
            names = ["coco_caption_karpathy_test"]
        super().__init__(*args, **kwargs, names=names, text_column_name='caption')
        print(f'CocoKarpathyDataset {split} len : {len(self)}')


class CocoKarpathyRecallDataset(CocoKarpathyBaseDataset):
    def __init__(self, *args, split='', names='', **kwargs):
        assert split in ['train', 'val', 'test']
        self.split = split
        if split == "train":
            names = ["coco_caption_karpathy_train", "coco_caption_karpathy_restval"]
        elif split == "val":
            names = ["coco_caption_karpathy_val"]
        elif split == "test":
            names = ["coco_caption_karpathy_test"]
        super().__init__(*args, **kwargs, names=names, text_column_name='caption')
        print(f'CocoKarpathyRecallDataset {split} len : {len(self)}')


    def get_text(self, image_index, text_key='caption'):
        texts = self.all_texts[image_index]
        encodings = self.tokenizer(
            texts,
            padding='max_length',
            max_length=self.max_text_len,
            truncation=True,
            return_special_tokens_mask=True,
            return_tensors='pt'
        )

        return {
            'text': None,
            'text_encodings': encodings,
            'text_index': None,
            'text_list': texts,
            'text_list_index': [image_index] * len(texts)
        }


