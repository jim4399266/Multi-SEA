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
from subsrc.datamodules import build_datamodule
from subsrc.modules import build_model

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
        'queue_size': config['queue_size'],
        'topk': config['top_k'],
        'epoch': config["max_epoch"],
        'lr': config["optimizer"]["init_lr"],
        'layer': config['num_top_layer'],
        'from_': '',
    }
    if config['attention_groups']:
        prefix_dict.update({'arch': f'{config["arch"]}_GQA_{config["attention_groups"]}'})
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
    print('-------------\n',log_name, '\n----------------------')
    modelsummary_callback = pl.callbacks.ModelSummary(max_depth=1)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_dir / f"version_{logger.version}",
        filename='step{step}-val_score{val/the_metric:.4f}',
        auto_insert_metric_name=False,
        save_top_k=3,
        monitor=f'val/the_metric',
        mode='max',
        save_last=False,
        verbose=True,
        save_weights_only=True,
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks = [modelsummary_callback, checkpoint_callback, lr_callback]

    dm = build_datamodule(config)
    model = build_model(config)

    trainer = pl.Trainer(
        # resume_from_checkpoint=config['load_path'],
        logger=logger,
        log_every_n_steps=50,
        precision=config['precision'],
        # amp_backend='apex' if config['apex'] else "native",
        # amp_level=config['amp_level'] if config['apex'] else None,

        accelerator=config['accelerator'],
        devices=config['devices'],
        # gpus=config['gpus'],
        strategy=strategy,
        # strategy='ddp_find_unused_parameters_true',
        use_distributed_sampler=False,
        # enable_model_summary=True,
        # benchmark=True,
        max_epochs=config['max_epoch'],
        callbacks=callbacks,
        # gradient_clip_val=None if config['manual_optimization'] else config['max_grad_norm'],
        # accumulate_grad_batches=None if config['manual_optimization'] else grad_steps,
        gradient_clip_val=config['max_grad_norm'],
        accumulate_grad_batches=grad_steps,

        fast_dev_run=config.get('fast_dev_run', False),
        # limit_train_batches=config.get(config['limit_train_batches'], None),
        # limit_val_batches=config.get(config['limit_val_batches'], None),
        # limit_test_batches=config.get(config['limit_test_batches'], None),
        # limit_predict_batches=config.get(config['limit_predict_batches'], None),
        num_sanity_val_steps=config['num_sanity_val_steps'],
        val_check_interval=config.get('val_check_interval', None),
        check_val_every_n_epoch=config.get('check_val_every_n_epoch', None),

    )

    if args.test_only:
        trainer.test(model, datamodule=dm)
    elif args.evaluate:
        trainer.validate(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)
        weight_paths = list(trainer.checkpoint_callback.best_k_models.keys())
        # weight_paths = list(Path(checkpoint_callback.dirpath).rglob('*.[pc][tk][hp]*'))
        # weight_paths = list(Path('/home/tzj/codes/my_clip/outputs/'
        #                          'irtr_bs200_pbs50_epoch6_lr1e-05_is224_from_model_base/version_19/').rglob(
        #     '*.[pc][tk][hp]*'))
        print(weight_paths)
        for ckpt in weight_paths:
            trainer.test(model, datamodule=dm, ckpt_path=str(ckpt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', default='./subsrc/configs/retrieval_coco_baseline.yaml')
    parser.add_argument('--config', default='./subsrc/configs/retrieval_coco.yaml')
    parser.add_argument('--devices', default='')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    if args.devices != '':
        config['devices'] = eval(args.devices)
    if args.debug:
        config['train_dataset_len'] = int(5 * config['per_gpu_batch_size'])
        config['val_dataset_len'] = int(-1)
        # config['val_dataset_len'] = int(10 * config['per_gpu_batch_size'])
        config['test_dataset_len'] = int(10 * config['per_gpu_batch_size'])
        config['batch_size'] = config['per_gpu_batch_size']
        # config['fast_dev_run'] = 5
        config['shuffle'] = False
        config['num_workers'] = 0
        # config['max_epoch'] = 3
        config['debug'] = True
    config['optimizer']['betas'] = eval(config['optimizer']['betas'])
    main(args, config)