# import sys
# sys.path.append('..')

from ..datasets.coco_karpathy_dataset import CocoKarpathyDataset, CocoKarpathyRecallDataset
# from .base_datamodule import BaseDataModule
from .datamodule_base import BaseDataModule

class CocoKarpathyDataModule(BaseDataModule):
    '''
    只是选择数据集，构建方法在 BaseDataModule 中
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # 初始化数据集
    @property
    def train_dataset_cls(self):
        return CocoKarpathyDataset

    @property
    def val_dataset_cls(self):
        return CocoKarpathyRecallDataset

    @property
    def test_dataset_cls(self):
        return CocoKarpathyRecallDataset

    @property
    def dataset_name(self):
        return 'coco'
