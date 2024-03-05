from typing import List, Union

from .model_retrieval import (
    RetrievalModuleWithQueue,
    RetrievalModuleWithQueue_1,
    RetrievalModuleWithDoubleQueue,
    RetrievalModuleWithQueueShared
)

_models = {
    # "baseline": RetrievalModule,
    # "aformer":RetrievalModule,
    "aformer_queue":RetrievalModuleWithQueue,
    "aformer_swiglu_queue":RetrievalModuleWithQueue,
    "aformer_swiglu_double_queue_new":RetrievalModuleWithDoubleQueue,
    "aformer_swiglu_queue_new":RetrievalModuleWithQueue_1,
    "aformer3_swiglu_queue_new":RetrievalModuleWithQueue_1,
    "aformer4_swiglu_queue_new":RetrievalModuleWithQueue_1,
    "aformer4_swiglu_queue_shared":RetrievalModuleWithQueueShared,
}

def build_model(config):
    print('### building model. ###')
    arch = config['arch'].lower()
    if config['checkpoint'] == "":
        return _models[arch].from_pretrained(config)
    else:
        return _models[arch].from_checkpoint(config)

