'''
检索任务
'''
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import functools
from collections import defaultdict
import torch.distributed as dist
from .dist_utils import all_gather_with_grad, concat_all_gather

@torch.no_grad()
def val_irtr_encoding(pl_module, data_loader):
    text_embeds_all = []
    text_feats_all = []
    text_atts_all = []

    image_embeds_all = []
    image_feats_all = []
    image_atts_all = []

    for batch in tqdm(data_loader, desc="Computing text and image features..."):
        image_feats, image_embeds, image_atts = pl_module.encoding_image(batch)
        text_feats, text_embeds, text_atts = pl_module.encoding_text(batch)

        text_embeds_all.append(text_embeds)
        text_feats_all.append(text_feats)
        text_atts_all.append(text_atts)
        image_embeds_all.append(image_embeds)
        image_feats_all.append(image_feats)
        image_atts_all.append(image_atts)

    text_embeds_all = torch.cat(text_embeds_all, dim=0)
    text_feats_all = torch.cat(text_feats_all, dim=0)
    text_atts_all = torch.cat(text_atts_all, dim=0)
    image_embeds_all = torch.cat(image_embeds_all, dim=0)
    image_feats_all = torch.cat(image_feats_all, dim=0)
    image_atts_all = torch.cat(image_atts_all, dim=0)
    return [text_embeds_all, text_feats_all, text_atts_all,
            image_embeds_all, image_feats_all, image_atts_all]

@torch.no_grad()
def val_irtr_recall_sort(pl_module, vectors):
    text_embeds_all, text_feats_all, text_atts_all,\
        image_embeds_all, image_feats_all, image_atts_all = vectors
    config = pl_module.hparams.config
    device = text_feats_all.device
    # 粗排，筛选 top_k 个候选集
    sims_matrix = image_feats_all @ text_feats_all.t()
    score_matrix_i2t = torch.full((len(image_feats_all), len(text_feats_all)), -100.).to(device)

    num_devices = pl_module.trainer.world_size
    rank = pl_module.trainer.global_rank
    step = sims_matrix.size(0) // num_devices + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    # 精排，图片检索文本
    for i, sims in tqdm(enumerate(sims_matrix[start:end]), desc="image2text recalling"):
        topk_sim, topk_idx = sims.topk(k=config['top_k'], dim=0)
        image_embeds_input = image_embeds_all[start + i].repeat(config['top_k'], 1, 1).to(device)
        image_atts_input = image_atts_all[start + i].repeat(config['top_k'], 1, 1).to(device)
        image2text_output = pl_module.aformer(
            image_embeds_input,
            attention_mask=image_atts_input,
            encoder_hidden_states=text_embeds_all[topk_idx].to(device),
            encoder_attention_mask=text_atts_all[topk_idx].to(device),
            mode='i2t'
        )
        score_i2t = pl_module.itm_head(image2text_output.last_hidden_state[:, 0])
        score_i2t = score_i2t[:, 1]
        # score_i2t = pl_module.itm_head(image2text_output.last_hidden_state[:, 0])[:, 1]
        score_matrix_i2t[start + i, topk_idx] = score_i2t + topk_sim #TODO  topk_sim 远大于 score_t2i

    # 精排，文本检索图片
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(text_feats_all), len(image_feats_all)), -100.).to(device)

    num_devices = pl_module.trainer.world_size
    rank = pl_module.trainer.global_rank
    step = sims_matrix.size(0) // num_devices + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in tqdm(enumerate(sims_matrix[start:end]), desc='text2image recalling'):
        topk_sim, topk_idx = sims.topk(k=config['top_k'], dim=0)
        text_embeds_input = text_embeds_all[start + i].repeat(config['top_k'], 1, 1).to(device)
        text_atts_input = text_atts_all[start + i].repeat(config['top_k'], 1, 1).to(device)

        text2image_output = pl_module.aformer(
            text_embeds_input,
            attention_mask=text_atts_input,
            encoder_hidden_states=image_embeds_all[topk_idx].to(device),
            encoder_attention_mask=image_atts_all[topk_idx].to(device),
            mode='t2i'
        )
        score_t2i = pl_module.itm_head(text2image_output.last_hidden_state[:, 0])[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score_t2i + topk_sim  #TODO  topk_sim 远大于 score_t2i


    # 多卡情况下，同步进度
    if pl_module.trainer.world_size > 1:
        torch.distributed.barrier()
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


# @torch.no_grad()
# def val_irtr(pl_module, vectors):
#     if '1k' in pl_module.hparams.config['coco_scale']:
#         # 使用mscoco的1k测试集
#
#
#     if '5k' in pl_module.hparams.config['coco_scale']:
#         # 使用mscoco的5k测试集
#         score_val_i2t, score_val_t2i = val_irtr_recall_sort(pl_module, vectors)
#         val_result = calculate_score(score_val_i2t, score_val_t2i, data_loader.dataset.index_mapper)
#         for item in ['txt_r1', 'txt_r5', 'txt_r10', 'txt_r_mean', 'img_r1', 'img_r5', 'img_r10', 'img_r_mean', 'r_mean']:
#             self.logger.experiment.add_scalar(f"{phase}_{dataset}{sacle}/{item}", val_result[item], cur_step)
#     if pl_module.hparams.config['coco_scale'] == '':
#         # 使用默认数据集（mscoco5k）


@torch.no_grad()
def calculate_score(scores_i2t, scores_t2i, index_mapper):
    img2txt, txt2img = {}, {}
    for k, v in index_mapper.items():
        # t_idx = k         i_idx = v[0]
        txt2img[k] = v[0]
        img2txt.setdefault(v[0], []).append(k)

    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    print(f'Target text len:{len(ranks)}')
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])
    print(f'Target image len:{len(ranks)}')
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2
    r_sum = (tr1 + tr5 + tr10) + (ir1 + ir5 + ir10)

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean,
                   'r_sum': r_sum}
    return eval_result