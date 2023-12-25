from .base_dataset import Flickr30KBaseDataset

class Flickr30KDataset(Flickr30KBaseDataset):
    def __init__(self, *args, split='', names='', **kwargs):
        assert split in ['train', 'val', 'test']
        self.split = split
        if split == "train":
            names = ["f30k_caption_karpathy_train", "f30k_caption_karpathy_val"]
        elif split == "val":
            names = ["f30k_caption_karpathy_test"]
        elif split == "test":
            names = ["f30k_caption_karpathy_test"]
        super().__init__(*args, **kwargs, names=names, text_column_name='caption')
        print(f'Flickr30kKarpathyDataset {split} len : {len(self)}')

class Flickr30KRecallDataset(Flickr30KBaseDataset):
    def __init__(self, *args, split='', names='', **kwargs):
        assert split in ['train', 'val', 'test']
        self.split = split
        if split == "train":
            names = ["f30k_caption_karpathy_train", "f30k_caption_karpathy_val"]
        elif split == "val":
            names = ["f30k_caption_karpathy_test"]
        elif split == "test":
            names = ["f30k_caption_karpathy_test"]
        super().__init__(*args, **kwargs, names=names, text_column_name='caption')
        print(f'Flickr30kKarpathyRecallDataset {split} len : {len(self)}')

    def get_text(self, image_index):
        # 测试时，返回一张图片对应的一组文本
        texts = self.all_texts[image_index]
        encodings = self.tokenizer(
            texts,  # 这里是一个列表，包含每张图片对应的一组文本
            padding='max_length',
            max_length=self.max_text_len,
            truncation=True,
            return_special_tokens_mask=True,  # 遮住特殊token的mask
            return_tensors='pt'
        )
        # 注意区分key中的text和cap，在collate会有不同处理
        return {
            'text': None,  # 正例的原文
            'text_encodings': encodings,  # 所有文本的encoding
            'text_index': None,  # 正例的下标
            'text_list': texts,  # 图片对应的文本列表
            'text_list_index': [image_index] * len(texts)
        }