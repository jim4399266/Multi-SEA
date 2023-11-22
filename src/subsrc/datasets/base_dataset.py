import random
import torch
from torch.utils.data import Dataset
import io
import pyarrow as pa
import os
from pathlib import Path
from typing import Union, Optional, List, Dict
from PIL import Image
from collections import defaultdict
from transformers import ViTImageProcessor, ViTFeatureExtractor
# import sys
# sys.path.append('..')
from ..transforms import keys_to_transforms

class CocoKarpathyBaseDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            transform_keys: List[str],
            image_size: int,
            names: List[str],
            text_column_name: str = "",
            remove_duplicate=True,
            max_text_len=40,
            image_only=False,
            tokenizer=None,
            dataset_len=-1,
    ):
        '''
        :param data_dir: where dataset file *.arrow lives; existence should be guaranteed via prepare_data.write_karpathy.py
        :param transform_keys: keys for generating augmented views of images
        :param image_size:
        :param names: prefix of '.arrow' file
        :param text_column_name: pyarrow table column name that has list of strings as elements
        :param remove_duplicate:
        :param dataset_len: 想要设置数据集的大小，用于测试

        用于读取经过预处理的MSCOCO数据集，并做统一的文本分词和图像缩放
        '''
        assert len(transform_keys) > 0
        super().__init__()
        self.image_size = image_size
        self.transforms = keys_to_transforms(transform_keys, size=image_size, )
        # 是否用到clip的transform
        self.clip_transform = False
        for key in transform_keys:
            if 'clip' in key:
                self.clip_transform = True
                break
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.image_only = image_only
        self.data_dir = data_dir
        self.tokenizer = tokenizer

        ############################# 读取.arrow文件 ##############################
        self.all_texts = list()
        if len(names) > 0:
            # tables包含训练集（验证集、测试集）
            tables = [
                pa.ipc.RecordBatchFileReader(
                    pa.memory_map(f'{data_dir}/{name}.arrow', 'r')
                ).read_all()
                for name in names if Path(f'{data_dir}/{name}.arrow').is_file()
            ]

            self.table_names = list()
            for i, name in enumerate(names):
                self.table_names += [name] * len(tables[i])

            # table包含4列： ['image', 'caption', 'image_id', 'split']
            # 分别是：      arrow格式的图片、标题、图片文件名、所属集合（train, val)
            # len(table) 是图片的数量
            self.table = pa.concat_tables(tables, promote=True)
            if dataset_len >= 0:  # 对数据集进行截断，用于小规模测试
                self.table = self.table[:dataset_len]
            if text_column_name != '':
                self.text_column_name = text_column_name
                self.all_texts = self.table[text_column_name].to_pandas().tolist()
                if isinstance(self.all_texts[0][0], str):
                    if remove_duplicate:
                        self.all_texts = [list(set(texts)) for texts in self.all_texts]
                else:  # snli
                    self.all_texts = [[t[1].strip() for t in texts] for texts in self.all_texts]

        # 构建index与样本的映射，如
        # {0: (0,0), 1: (0,1), 2: (0,2), 3: (0,3), 4: (0,4), 5: (1,0) ......}
        # key 代表每一对数据的全局下标，
        # value元组中，第一位代表图片的下标，
        # 第二位代表图片对应标题的下标（例：每张图片有5个标题，那么第二位就是0，1，2，3，4这5个数
        self.index_mapper = dict()
        self.image_mapper = defaultdict(list)

        if text_column_name != '' and not self.image_only:
            j = 0
            all_texts = self.all_texts[:dataset_len] if dataset_len >= 0 else self.all_texts
            for i, texts in enumerate(all_texts):
                # # 统一每张图片对应5条文本，去除多余文本
                # if len(texts) > 5:
                #     print(i, "--", len(texts))
                #     texts = texts[:5]
                #     self.all_texts[i] = texts
                # elif len(texts) < 5:
                #     print(i, "--", len(texts))
                for _j in range(len(texts)):
                    # 构建文本和图片的映射关系：第j段文本是第i张图片中的第_j个文本（_j： 0~4）
                    self.image_mapper[i].append(j)
                    self.index_mapper[j] = [i, _j]
                    j += 1
        # 如果没有文本，则只有图片的下标
        else:
            for i in range(len(self.table)):
                self.index_mapper[i] = (i, None)

    def __len__(self):
        return len(self.table)


    def __getitem__(self, index):
        # 在训练时，只需返回一张图片和一段文本的图文对
        # 测试时，返回一张图片对应的一组文本
        ret = dict()
        try:
            ret.update(self.get_image(index))
            if not self.image_only:
                # 原始get_text()方法中，随机抽取一段文本对应图片
                text = self.get_text(index)
                ret.update(text)
        except Exception as e:
            print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
        return ret

    @property
    def corpus(self):
        # 返回文本数据
        return [text for texts in self.all_texts for text in texts]

    def get_raw_image(self, image_index, image_key='image'):
        # 读取原始图片
        # index, caption_index = self.index_mapper[raw_index]
        image_bytes = io.BytesIO(self.table[image_key][image_index].as_py())
        image_bytes.seek(0)
        if self.clip_transform:
            return Image.open(image_bytes).convert('RGBA')
        else:
            return Image.open(image_bytes).convert('RGB')

    def get_image(self, image_index, image_key='image'):
        image = self.get_raw_image(image_index, image_key)
        # 利用transform转换为统一格式
        if self.clip_transform:
            image_tensor = [tr(image) for tr in self.transforms]
        else:
            image_tensor = [tr(image, return_tensors="pt")['pixel_values'].squeeze() for tr in self.transforms]

        return {
            'image': image_tensor,
            'image_index': image_index,
        }

    def get_text(self, image_index, text_key='caption'):
        texts = self.all_texts[image_index]  # 每张图片对应的一组文本
        text_id = random.choice(range(len(texts)))
        encodings = self.tokenizer(
            texts[text_id],
            padding='max_length',
            add_special_tokens=True,
            max_length=self.max_text_len,
            truncation=True,
            return_special_tokens_mask=True,  # 遮住特殊token的mask
            return_tensors='pt',
        )
        # 注意区分key中的text和cap，在collate会有不同处理
        return {
            'text': texts[text_id],  # 正例的原文
            'text_encodings': encodings,  # 正例的encoding
            'text_index': (image_index, text_id),  # 正例的下标
            'text_list': texts,  # 图片对应的文本列表
            'text_list_index': [image_index] * len(texts)
        }

    def collate(self, batch, mlm_collator=None):
        keys = set([key for b in batch for key in b.keys()])
        # 将batch中属于同一个keys的信息放到一起
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        # ==================================== 整理图片 ====================================
        # 取出与image相关的keys
        img_keys = ['image']
        # 得到每个图片tensor的shape
        # for img_key in img_keys:
        #     img = dict_batch[img_key]
        #     for i in img :
        #         if i is not None:
        #             for j in i:
        #                 # 检查图片维度
        #                 assert (len(j.shape) == 3
        #                         ), f"Collate error, an image should be in shape of (3, H, W), instead of given {j.shape}"

        # 将所有图片都扩大到[3, max_height, max_width] 个像素，
        # 并且将dict_batch中的list转换为tensor
        for img_key in img_keys:
            imgs = [img[0] for img in dict_batch[img_key]]
            new_images = torch.stack(imgs, dim=0)
            dict_batch[img_key] = new_images
        dict_batch['image_index'] = torch.tensor(dict_batch['image_index'], dtype=torch.long)

        # ==================================== 整理文本 ====================================
        encodings = {}
        e_keys = set([key for b in dict_batch['text_encodings'] for key in b.keys()])
        for k in e_keys:
            encodings[k] = torch.cat([dic[k] if k in dic else None for dic in dict_batch['text_encodings']], dim=0)
        dict_batch['text_encodings'] = encodings
        text_list_index = [i for index in dict_batch['text_list_index'] for i in index]
        dict_batch['text_list_index'] = torch.tensor(text_list_index, dtype=torch.long)
        # try:
        #     text_list_index = [i for index in dict_batch['text_list_index'] for i in index]
        #     dict_batch['text_list_index'] = torch.tensor(text_list_index, dtype=torch.long)
        # except:
        #     print('====================',dict_batch['text_list'])
        #     print('====================',dict_batch['text_list_index'])
        #     exit()

        return dict_batch
