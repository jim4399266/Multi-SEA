from typing import List, Union

from .model_retrieval import RetrievalModuleWithQueue_1


_models = {
    "aformer_swiglu_queue_new":RetrievalModuleWithQueue_1,
    "aformer3_swiglu_queue_new":RetrievalModuleWithQueue_1}

def build_model(config):
    print('### building model. ###')
    arch = config['arch'].lower()
    if config['checkpoint'] == "":
        return _models[arch].from_pretrained(config)
    else:
        return _models[arch].from_checkpoint(config)

