# Multi-SEA: Multi-stage Semantic Enhancement and Aggregation for Image-Text Retrieval
The paper was accepted on 24 March 2025 and available online 14 April 2025. Information Processing and Management 62 (2025) 104165. [https://doi.org/10.1016/j.ipm.2025.104165](https://doi.org/10.1016/j.ipm.2025.104165)

## Introduction
We propose a Multi-stage Semantic Enhancement and Aggregation framework (**Multi-SEA**) with novel networks and training schemes. Multi-SEA first designs a fusion module with agent attention and gating mechanism. It enhances uni-modal information and aggregates fine-grained cross-modal information by involving different stages. Multi-SEA then introduces a three-stage scheme to integrate the two structures mentioned above together. 

<img src="figs/model_structure.png" width="600"> 

Eventually, Multi-SEA utilizes a negative sample queue and hierarchical scheme to facilitate robust contrastive learning and promote expressive capabilities from implicit information. Experimental results demonstrate that Multi-SEA outperforms the state-of-the-art schemes with a large margin.

<img src="figs/pipeline.png" width="600"> 

## Backbone pretrained models
We employ the Roberta-base model and Vit-B/16 model as the backbones to preliminarily encode our raw data, which can be found in following links quickly.

| Visual Backbone | Text Backbone |
|------------------------|------------------------|
| [vit-b-16](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) | [Roberta-base](https://drive.google.com/file/d/1ddE0BSYxvdQLAH0t6fTk5UCV1a9B6q7x/view?usp=sharing) | 

## Multi-SEA checkpoints
We provide the tensorboard logs and checkpoints fine-tuned on Flickr30k and MSCOCO. The checkpoints  contain weights, configs, optimizers, and other training information saved by Pytorch Lightning.
| Checkpoint on Flickr30k | Checkpoint on MSCOCO |
|------------------------|------------------------|
| [Multi-SEA_flickr30k](https://drive.google.com/file/d/1rkh7GWXarUBQsK9JocTK9RUCnEMJP1HQ/view?usp=sharing) | [Multi-SEA_mscoco](https://drive.google.com/file/d/1lBSAJH477P9aOFkfZG6Lyw8DIc02f1tw/view?usp=sharing) | 


## Requirements
* Python version >= 3.9.0

* [PyTorch](https://pytorch.org/) version >= 2.0.0

* Install other libraries via
```
pip install -r requirements.txt
```

## Datasets
We follow [ViLT](https://github.com/dandelin/ViLT) and use `pyarrow` to serialize the datasets. See [this link](https://github.com/dandelin/ViLT/blob/master/DATA.md) for details.


## Training
### On Flickr30K
We should find the config file in <ins>src/subsrc/configs/retrieval_flickr30k.yaml

then replace the following file path: 
* data_root : the directory of your dataset
* vit :  the directory of image backbone
* tokenizer : the directory of text backbone

finally run the script in /src :
```
python main.py --config=./subsrc/configs/retrieval_flickr30k.yaml   --devices=[0] 
```

### On MSCOCO
We should find the config file in <ins>src/subsrc/configs/retrieval_coco.yaml

then replace the following file path: 
* data_root : the directory of your dataset
* vit :  the directory of image backbone
* tokenizer : the directory of text backbone

finally run the script in /src :
```
python main.py --config=./subsrc/configs/retrieval_coco.yaml   --devices=[0] 
```

## Testing
### On Flickr30K
We should find the config file in <ins>src/subsrc/configs/retrieval_flickr30k.yaml

then replace the following file path: 
* data_root : the directory of your dataset
* test_checkpoints_dir : the directory of checkpoint
* vit :  the directory of image backbone
* tokenizer : the directory of text backbone

finally run the script in /src :
```
python main.py --config=./subsrc/configs/retrieval_flickr30k.yaml   --devices=[0] --test_only
```

### On MSCOCO
We should find the config file in <ins>src/subsrc/configs/retrieval_coco.yaml

then replace the following file path: 
* data_root : the directory of your dataset
* test_checkpoints_dir : the directory of checkpoint
* vit :  the directory of image backbone
* tokenizer : the directory of text backbone

finally run the script in /src :
```
python main.py --config=./subsrc/configs/retrieval_coco.yaml   --devices=[0] --test_only
```

## Citation
If you use our work, please cite:
```
@article{tian2025multi,
  title={Multi-SEA: Multi-stage Semantic Enhancement and Aggregation for image--text retrieval},
  author={Tian, Zijing and Ou, Zhonghong and Zhu, Yifan and Lyu, Shuai and Zhang, Hanyu and Xiao, Jinghua and Song, Meina},
  journal={Information Processing \& Management},
  volume={62},
  number={5},
  pages={104165},
  year={2025},
  issn = {0306-4573},
  doi = {https://doi.org/10.1016/j.ipm.2025.104165},
  url = {https://www.sciencedirect.com/science/article/pii/S0306457325001062},
  publisher={Elsevier}
}
```
## Acknowledgement

The implementation of Mulit-SEA relies on resources from [Bert(pytorch)](https://github.com/codertimo/BERT-pytorch), [CLIP](https://github.com/openai/CLIP), [llama](https://github.com/meta-llama/llama), and [Pytorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) . We thank the original authors for their open-sourcing.

