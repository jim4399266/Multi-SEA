import argparse
import sys
import torch
import yaml
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import pytorch_lightning as pl

from vit_rollout import VITAttentionRollout, PredictBaseDataset
from vit_grad_rollout import VITAttentionGradRollout

sys.path.append('../')
from modules import build_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./explain_flickr30k.yaml')
    # parser.add_argument('--devices', default='')

    # parser.add_argument('--use_cuda', action='store_true', default=False,
    #                     help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='../examples/16151663.jpg',
                        help='Input image path')
    parser.add_argument('--txt', type=str, default='two women',
                        help='Input text')
    parser.add_argument('--head_fusion', type=str, default='min',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    parser.add_argument('--attention_mode', type=str, default='i2t',
                        help='Attention mode. '
                             'Can be "i2t", "t2i", "i2i_p1", "i2i_p2", "t2t_p1", "t2t_p2"')
    args = parser.parse_args()
    # args.use_cuda = args.use_cuda and torch.cuda.is_available()
    # if args.use_cuda:
    #     print("Using GPU")
    # else:
    #     print("Using CPU")
    return args


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def get_input_data(args, transform, tokenizer, device='cpu'):
    img = Image.open(args.image_path)
    img = img.resize((224, 224))
    # txt = 'The pilot looking out of a British Airways airplane while two men are standing outside.'
    txt = args.txt
    dataset = PredictBaseDataset(txt, img, transform, tokenizer)

    data_loader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate)
    return img, txt, data_loader


if __name__ == '__main__':
    args = get_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    args.use_cuda = True if config["accelerator"] == 'gpu' else False
    device = 'cuda:' + str(config['devices'][0]) if args.use_cuda else 'cpu'
    model = build_model(config)
    # model = torch.hub.load('facebookresearch/deit:main',
    #     'deit_tiny_patch16_224', pretrained=True)
    model.eval()
    model = model.to(device)
    trainer = pl.Trainer(
        logger=False,
        accelerator=config['accelerator'],
        devices=config['devices'],
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    tokenizer = AutoTokenizer.from_pretrained(config['text_encoder_config']['tokenizer'])
    img, txt, data_loader = get_input_data(args, transform, tokenizer, device)

    img_tag = args.image_path.split("/")[-1].split(".")[0]
    if args.category_index is None:
        print("Doing Attention Rollout")
        attention_rollout = VITAttentionRollout(trainer,model, head_fusion=args.head_fusion,
            discard_ratio=args.discard_ratio)
        mask = attention_rollout(data_loader, mode=args.attention_mode)
        np_img = np.array(img)[:, :, ::-1]
        for i, _mask in enumerate(mask, start=1):
            name = "{}_attention_rollout_layer{}_{}_{:.3f}_{}:{}.png".format(
            img_tag, i, args.attention_mode, args.discard_ratio, args.head_fusion, txt)

            _mask = cv2.resize(_mask, (np_img.shape[1], np_img.shape[0]))
            _mask = show_mask_on_image(np_img, _mask)
            cv2.imwrite(name, _mask)

    else:
        print("Doing Gradient Attention Rollout")
        grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
        mask = grad_rollout(data_loader, args.category_index)
        name = "{}_grad_rollout_{}_{}_{:.3f}_{}.png".format(img_tag, args.attention_mode, args.category_index,
            args.discard_ratio, args.head_fusion)


    # np_img = np.array(img)[:, :, ::-1]
    # mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    # mask = show_mask_on_image(np_img, mask)
    # # cv2.imshow("Input Image", np_img)
    # # cv2.imshow(name, mask)
    # cv2.imwrite("input.png", np_img)
    # cv2.imwrite(name, mask)
    # # cv2.waitKey(-1)