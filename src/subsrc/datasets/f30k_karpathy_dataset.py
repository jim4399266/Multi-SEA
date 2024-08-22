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