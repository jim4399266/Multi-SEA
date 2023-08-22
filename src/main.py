import os
import copy
import torch
import argparse
import ruamel.yaml as yaml

import pytorch_lightning.loggers
import pytorch_lightning as pl
from pathlib import Path

# from datamodules.multitask_datamodule import MTDataModule

# from modules.blip_module import BLIPModule
from datamodules import build_datamodule
from modules import build_model

os.environ["NCCL_DEBUG"] = "INFO"
torch.set_float32_matmul_precision('high')

def main(args, config):
    # 如果不用GPU，则num_gpus=0，防止下面除0，num_gpus置为1
    config['num_device'] = config['devices'] if isinstance(config['devices'], int) else len(config['devices'])
    config['dist'] = True if config['num_device'] > 1 else False
    strategy = 'ddp_find_unused_parameters_true' if config['num_device'] > 1 else 'auto'
    grad_steps = max(
        config['batch_size'] //
        (config['per_gpu_batch_size'] * max(1, config['num_device']) * config['num_nodes'])
        , 1)
    config['gradient_accumulation_steps'] = grad_steps

    log_dir = config['log_dir']
    prefix_dict = {
        'task': '-'.join((list(config['task_name'].keys()))),
        'arch': config['arch'],
        'bs': config["batch_size"],
        'pbs': config["per_gpu_batch_size"],
        'epoch': config["max_epoch"],
        'lr': config["optimizer"]["init_lr"],
        'layer': config['num_top_layer'],
        'from_': '',
    }
    if config['pretrained'] == "":
        prefix_dict.update(
            {'from_': f'{config["image_encoder_config"]["vit_name"]}_'
                      f'{config["image_encoder_config"]["image_size"]}_'
                      f'{config["text_encoder_config"]["tokenizer_name"]}'}
        )
        log_name = '_'.join([f'{k}{v}' if (k != 'task' and k != 'arch') else f'{v}'
                             for k, v in prefix_dict.items()])
    else:
        prefix_dict.update(
            {'from_': f'{config["pretrained"].split("/")[-1].split(".")[0]}'}
        )
        log_name = '_'.join([f'{k}{v}' if (k != 'task' and k != 'arch') else f'{v}'
                             for k, v in prefix_dict.items()])

    output_dir = config['output_dir']
    if output_dir != None or "" or '':
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_dir = Path(output_dir) / log_name

    logger = pl.loggers.TensorBoardLogger(
        save_dir=log_dir,
        name=log_name,
        # default_hp_metric=False,    # 禁用 PyTorch Lightning 默认的 hparams 评估指标, 启用 TensorboardX
    )

    modelsummary_callback = pl.callbacks.ModelSummary(max_depth=1)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_dir / f"version_{logger.version}",
        filename='step{step}-val_score{val/' + f'{prefix_dict["task"]}' + '/r_mean:.4f',

    )