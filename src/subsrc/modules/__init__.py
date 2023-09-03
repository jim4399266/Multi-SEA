from typing import List, Union

from .model_retrieval import RetrievalModule

_models = {
    # "f30k": F30KCaptionKarpathyDataModule,
    "baseline": RetrievalModule,
    "aformer":RetrievalModule,
    # "aformer": RetrievalMomentumModule
}

def build_model(config):
    print('### building model. ###')
    arch = config['arch'].lower()
    return _models[arch].from_pretrained(config)

