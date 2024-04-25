from typing import List, Union

from .model_retrieval import (
    RetrievalModule,
    RetrievalModuleWithQueue,
    RetrievalModuleWithQueue_1,
    RetrievalModuleWithDoubleQueue,
    RetrievalModuleWithQueueShared,
    RetrievalModuleWithQueueContrast1,
    RetrievalModuleWithQueueContrast2,
    RetrievalModuleWithQueueContrast3
)

_models = {
    # "baseline": RetrievalModule,
    "aformer4_swiglu":RetrievalModule,
    "aformer_queue":RetrievalModuleWithQueue,
    "aformer_swiglu_queue":RetrievalModuleWithQueue,
    "aformer_swiglu_double_queue_new":RetrievalModuleWithDoubleQueue,
    "aformer_swiglu_queue_new":RetrievalModuleWithQueue_1,
    "aformer3_swiglu_queue_new":RetrievalModuleWithQueue_1,
    "aformer4_swiglu_queue_new":RetrievalModuleWithQueue_1,
    "aformer4_swiglu_queue_shared":RetrievalModuleWithQueueShared,
    "aformer4_swiglu_queue_contrast1":RetrievalModuleWithQueueContrast1,
    "aformer4_swiglu_queue_contrast2":RetrievalModuleWithQueueContrast2,
    "aformer4_swiglu_queue_contrast3":RetrievalModuleWithQueueContrast3,
}

def build_model(config):
    print('### building model. ###')
    arch = config['arch'].lower()
    if config['checkpoint'] == "":
        return _models[arch].from_pretrained(config)
    else:
        return _models[arch].from_checkpoint(config)

