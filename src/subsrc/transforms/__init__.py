from transformers import ViTImageProcessor, ViTFeatureExtractor

from .transform import (
    pixelbert_transform,
    pixelbert_transform_randaug,
    vit_transform,
    vit_transform_randaug,
    imagenet_transform,
    imagenet_transform_randaug,
    clip_transform,
    clip_transform_randaug,
)

_transforms = {
    "pixelbert": pixelbert_transform,
    "pixelbert_randaug": pixelbert_transform_randaug,
    "vit": vit_transform,
    "vit_randaug": vit_transform_randaug,
    "imagenet": imagenet_transform,
    "imagenet_randaug": imagenet_transform_randaug,
    "clip": clip_transform,
    "clip_randaug": clip_transform_randaug,
}


def keys_to_transforms(keys: list, size=224):
    trans = []
    for key in keys:
        if key not in _transforms:
            trans.append(ViTImageProcessor.from_pretrained(key))
        else:
            trans.append(_transforms[key](size=size))
    return trans
    # return [_transforms[key](size=size) for key in keys]
