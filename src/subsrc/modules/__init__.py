from typing import List, Union

from .model_retrieval import RetrievalModuleWithQueue,RetrievalModuleWithQueue_1

_models = {
    # "f30k": F30KCaptionKarpathyDataModule,
    # "baseline": RetrievalModule,
    # "aformer":RetrievalModule,
    "aformer_queue":RetrievalModuleWithQueue,
    "aformer_swiglu_queue":RetrievalModuleWithQueue,
    "aformer_swiglu_queue_new":RetrievalModuleWithQueue_1,
    # "aformer": RetrievalMomentumModule
}

def build_model(config):
    print('### building model. ###')
    arch = config['arch'].lower()
    if config['checkpoint'] == "":
        return _models[arch].from_pretrained(config)
    else:
        return _models[arch].from_checkpoint(config)

