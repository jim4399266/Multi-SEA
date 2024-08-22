from typing import List, Union



from .coco_karpathy_datamodule import CocoKarpathyDataModule
from .flickr30k_karpathy_datamodule import Flickr30kKarpathyDataModule


_datamodules = {
    "flickr30k": Flickr30kKarpathyDataModule,
    "coco": CocoKarpathyDataModule,
}

def build_datamodule(config):
    print('### building datamodule. ###')
    dataset = config['datasets']
    if isinstance(dataset, List) and len(dataset) > 1:
        return None
    elif isinstance(dataset, List) and len(dataset) == 1:
        return _datamodules[dataset[0]](config)
    elif isinstance(dataset, str):
        return _datamodules[dataset](config)

