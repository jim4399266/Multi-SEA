from ..datasets.f30k_karpathy_dataset import Flickr30KDataset, Flickr30KRecallDataset
from .datamodule_base import BaseDataModule

class Flickr30kKarpathyDataModule(BaseDataModule):
    '''
    只是选择数据集，构建方法在 BaseDataModule 中
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # 初始化数据集
    @property
    def train_dataset_cls(self):
        return Flickr30KDataset

    @property
    def val_dataset_cls(self):
        return Flickr30KRecallDataset

    @property
    def test_dataset_cls(self):
        return Flickr30KRecallDataset

    @property
    def dataset_name(self):
        return 'flickr30k'
