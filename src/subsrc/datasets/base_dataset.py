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

        '''
        assert len(transform_keys) > 0
        super().__init__()
        self.image_size = image_size
        self.transforms = keys_to_transforms(transform_keys, size=image_size)


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

        #############################  read '.arrow'  files ##############################
        self.all_texts = list()
        if len(names) > 0:
            tables = [
                pa.ipc.RecordBatchFileReader(
                    pa.memory_map(f'{data_dir}/{name}.arrow', 'r')
                ).read_all()
                for name in names if Path(f'{data_dir}/{name}.arrow').is_file()
            ]

            self.table_names = list()
            for i, name in enumerate(names):
                self.table_names += [name] * len(tables[i])

            self.table = pa.concat_tables(tables, promote=True)
            if dataset_len >= 0:
                self.table = self.table[:dataset_len]
            if text_column_name != '':
                self.text_column_name = text_column_name
                self.all_texts = self.table[text_column_name].to_pandas().tolist()
                if isinstance(self.all_texts[0][0], str):
                    if remove_duplicate:
                        self.all_texts = [list(set(texts)) for texts in self.all_texts]
                else:  # snli
                    self.all_texts = [[t[1].strip() for t in texts] for texts in self.all_texts]

        # Construct the mapping of index to samples, as
        # {0: (0,0), 1: (0,1), 2: (0,2), 3: (0,3), 4: (0,4), 5: (1,0) ......}
        # key  represents the global index of each pair of data,
        # The first element in the value tuple is the index of the image,
        # The second element represents the caption index of the image (e.g., each image has 5 captions, so the second element is 0,1,2,3,4)

        self.index_mapper = dict()
        self.image_mapper = defaultdict(list)

        if text_column_name != '' and not self.image_only:
            j = 0
            all_texts = self.all_texts[:dataset_len] if dataset_len >= 0 else self.all_texts
            for i, texts in enumerate(all_texts):
                for _j in range(len(texts)):
                    # Build a mapping between text and image: the JTH text is the _jth text in the ith image (_j: 0-4)
                    self.image_mapper[i].append(j)
                    self.index_mapper[j] = [i, _j]
                    j += 1
        # If there is no text, there is only the index of the image
        else:
            for i in range(len(self.table)):
                self.index_mapper[i] = (i, None)

    def __len__(self):
        return len(self.table)


    def __getitem__(self, index):
        # At training time, we simply return an image-text pair of an image and a piece of text
        # When testing, return a set of text for an image
        ret = dict()
        try:
            ret.update(self.get_image(index))
            if not self.image_only:

                text = self.get_text(index)
                ret.update(text)
        except Exception as e:
            print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
        return ret

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def get_raw_image(self, image_index, image_key='image'):
        image_bytes = io.BytesIO(self.table[image_key][image_index].as_py())
        image_bytes.seek(0)
        if self.clip_transform:
            return Image.open(image_bytes).convert('RGBA')
        else:
            return Image.open(image_bytes).convert('RGB')

    def get_image(self, image_index, image_key='image'):
        image = self.get_raw_image(image_index, image_key)
        if self.clip_transform:
            image_tensor = [tr(image) for tr in self.transforms]
        else:
            image_tensor = [tr(image, return_tensors="pt")['pixel_values'].squeeze() for tr in self.transforms]

        return {
            'image': image_tensor,
            'image_index': image_index,
        }

    def get_text(self, image_index, text_key='caption'):
        texts = self.all_texts[image_index]
        text_id = random.choice(range(len(texts)))
        encodings = self.tokenizer(
            texts[text_id],
            padding='max_length',
            add_special_tokens=True,
            max_length=self.max_text_len,
            truncation=True,
            return_special_tokens_mask=True,
            return_tensors='pt',
        )

        return {
            'text': texts[text_id],  # The original text of the positive example
            'text_encodings': encodings,  # The encoding of the positive example
            'text_index': (image_index, text_id),  # The index of the positive example
            'text_list': texts,  # A list of text corresponding to the image
            'text_list_index': [image_index] * len(texts)
        }

    def collate(self, batch, mlm_collator=None):
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        # ==================================== Organizing images ====================================
        img_keys = ['image']
        for img_key in img_keys:
            imgs = [img[0] for img in dict_batch[img_key]]
            new_images = torch.stack(imgs, dim=0)
            dict_batch[img_key] = new_images
        dict_batch['image_index'] = torch.tensor(dict_batch['image_index'], dtype=torch.long)

        # ==================================== Organizing text ====================================
        encodings = {}
        e_keys = set([key for b in dict_batch['text_encodings'] for key in b.keys()])
        for k in e_keys:
            encodings[k] = torch.cat([dic[k] if k in dic else None for dic in dict_batch['text_encodings']], dim=0)
        dict_batch['text_encodings'] = encodings
        text_list_index = [i for index in dict_batch['text_list_index'] for i in index]
        dict_batch['text_list_index'] = torch.tensor(text_list_index, dtype=torch.long)

        return dict_batch

class Flickr30KBaseDataset(Dataset):
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

        '''
        assert len(transform_keys) > 0
        super().__init__()
        self.image_size = image_size
        self.transforms = keys_to_transforms(transform_keys, size=image_size)

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


        self.all_texts = list()
        if len(names) > 0:

            tables = [
                pa.ipc.RecordBatchFileReader(
                    pa.memory_map(f'{data_dir}/{name}.arrow', 'r')
                ).read_all()
                for name in names if Path(f'{data_dir}/{name}.arrow').is_file()
            ]

            self.table_names = list()
            for i, name in enumerate(names):
                self.table_names += [name] * len(tables[i])


            self.table = pa.concat_tables(tables, promote=True)
            if dataset_len >= 0:
                self.table = self.table[:dataset_len]
            if text_column_name != '':
                self.text_column_name = text_column_name
                self.all_texts = self.table[text_column_name].to_pandas().tolist()
                if isinstance(self.all_texts[0][0], str):
                    if remove_duplicate:
                        self.all_texts = [list(set(texts)) for texts in self.all_texts]
                else:  # snli
                    self.all_texts = [[t[1].strip() for t in texts] for texts in self.all_texts]

        # Construct the mapping of index to samples, as
        # {0: (0,0), 1: (0,1), 2: (0,2), 3: (0,3), 4: (0,4), 5: (1,0) ......}
        # key  represents the global index of each pair of data,
        # The first element in the value tuple is the index of the image,
        # The second element represents the caption index of the image (e.g., each image has 5 captions, so the second element is 0,1,2,3,4)
        self.index_mapper = dict()
        self.image_mapper = defaultdict(list)

        if text_column_name != '' and not self.image_only:
            j = 0
            all_texts = self.all_texts[:dataset_len] if dataset_len >= 0 else self.all_texts
            for i, texts in enumerate(all_texts):
                for _j in range(len(texts)):
                    # Build a mapping between text and image: the JTH text is the _jth text in the ith image (_j: 0-4)
                    self.image_mapper[i].append(j)
                    self.index_mapper[j] = [i, _j]
                    j += 1
        # If there is no text, there is only the index of the image
        else:
            for i in range(len(self.table)):
                self.index_mapper[i] = (i, None)
    def __len__(self):
        return len(self.table)


    def __getitem__(self, index):
        # At training time, we simply return an image-text pair of an image and a piece of text
        # When testing, return a set of text for an image
        ret = dict()
        try:
            ret.update(self.get_image(index))
            if not self.image_only:
                text = self.get_text(index)
                ret.update(text)
        except Exception as e:
            print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
        return ret

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def get_raw_image(self, image_index, image_key='image'):
        image_bytes = io.BytesIO(self.table[image_key][image_index].as_py())
        image_bytes.seek(0)
        if self.clip_transform:
            return Image.open(image_bytes).convert('RGBA')
        else:
            return Image.open(image_bytes).convert('RGB')

    def get_image(self, image_index, image_key='image'):
        image = self.get_raw_image(image_index, image_key)
        if self.clip_transform:
            image_tensor = [tr(image) for tr in self.transforms]
        else:
            image_tensor = [tr(image, return_tensors="pt")['pixel_values'].squeeze() for tr in self.transforms]

        return {
            'image': image_tensor,
            'image_index': image_index,
        }

    def get_text(self, image_index, text_key='caption'):
        texts = self.all_texts[image_index]  #
        text_id = random.choice(range(len(texts)))
        encodings = self.tokenizer(
            texts[text_id],
            padding='max_length',
            add_special_tokens=True,
            max_length=self.max_text_len,
            truncation=True,
            return_special_tokens_mask=True,
            return_tensors='pt',
        )

        return {
            'text': texts[text_id],
            'text_encodings': encodings,
            'text_index': (image_index, text_id),
            'text_list': texts,
            'text_list_index': [image_index] * len(texts)
        }

    def collate(self, batch, mlm_collator=None):
        keys = set([key for b in batch for key in b.keys()])

        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        # ==================================== Organizing images  ====================================
        img_keys = ['image']
        for img_key in img_keys:
            imgs = [img[0] for img in dict_batch[img_key]]
            new_images = torch.stack(imgs, dim=0)
            dict_batch[img_key] = new_images
        dict_batch['image_index'] = torch.tensor(dict_batch['image_index'], dtype=torch.long)

        # ==================================== Organizing text  ====================================
        encodings = {}
        e_keys = set([key for b in dict_batch['text_encodings'] for key in b.keys()])
        for k in e_keys:
            encodings[k] = torch.cat([dic[k] if k in dic else None for dic in dict_batch['text_encodings']], dim=0)
        dict_batch['text_encodings'] = encodings
        text_list_index = [i for index in dict_batch['text_list_index'] for i in index]
        dict_batch['text_list_index'] = torch.tensor(text_list_index, dtype=torch.long)
        return dict_batch


