
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import functools
from collections import defaultdict
import torch.distributed as dist
from .dist_utils import all_gather_with_grad, concat_all_gather

def in_modality_g2l_loss(local_feature, global_feature, temp=1., attention_mask=None):
    '''
    :param global_feature: bs, dim
    :param local_feature: bs, len, dim
    :param temp: matmul temperature
    :param attention_mask: text attention mask: bs, len
    :return:
    '''
    FILL = float('-inf')
    global_feature = global_feature.unsqueeze(1)   # bs, 1, dim
    bs, local_len, dim = local_feature.size()
    # 正样本对应的 global_feature 和 local_feature 进行点积，越小越好
    logits_pos = torch.matmul(local_feature, global_feature.permute(0, 2, 1)) / temp # bs, len, 1
    # 对文本填充的部分进行遮蔽，遮蔽部分赋值 负无穷，去除 softmax 时的影响
    if attention_mask is not None:
        tmp_mask = attention_mask.unsqueeze(-1)
        logits_pos = logits_pos.masked_fill(tmp_mask != 1, FILL)

    # 接下来对所有负样本进行点积，越大越好
    # 每个样本的 global feature 需要与所有样本的 local feature 计算
    # 相当于每个正样本中的 global feature:[1, dim] 需要与 每个负样本中的每个token(bs * len)进行点积
    # 即  global_feature_n: bs, dim        local_feature_n: (bs * local_len), dim
    global_feature_n, local_feature_n = global_feature.reshape(-1, dim), local_feature.reshape(-1, dim)
    logits_neg = torch.matmul(global_feature_n, local_feature_n.T) / temp  # bs, (bs * local_len)
    logits_neg = logits_neg.reshape(bs, bs, local_len)  # bs, bs, local_len
    # 首先需要对正样本进行遮蔽
    tmp_mask = 1 - torch.eye(bs)[:, :, None].to(logits_neg.device)
    logits_neg = logits_neg.masked_fill(tmp_mask != 1, FILL)
    # 对文本填充的部分进行遮蔽，遮蔽部分赋值 负无穷，去除 softmax 时的影响
    if attention_mask is not None:
        tmp_mask = attention_mask.unsqueeze(0)   # 1, bs, len
        logits_neg = logits_neg.masked_fill(tmp_mask != 1, FILL)

    # 每个正样本的得分需要与所有其他负样本得分进行 softmax 计算，要使 正样本的得分 占比更大
    # 正样本有 local_len 个，负样本有 (bs * local_len) 个
    # 因此需要将负样本进行维度变换:
    logits_neg = logits_neg.reshape(bs, -1).unsqueeze(1).expand(-1, local_len, -1) # bs, local_len, (bs * local_len)

    # Since this is multiclass, we concat the positive along the class dimension before performing log softmax.
    pred_lgt = torch.cat([logits_pos, logits_neg], dim=-1)
    pred_log = F.log_softmax(pred_lgt, dim=-1)

    # The positive score is the first element of the log softmax
    if attention_mask is not None:
        pred_log = -pred_log[:, :, 0].squeeze()
        pred_log = pred_log.masked_fill(attention_mask != 1, 0.)
        loss = (torch.sum(pred_log, dim=1) / torch.sum(attention_mask, dim=1)).mean()
    else:
        pred_log = -pred_log[:, :, 0]
        loss = pred_log.mean()
    return loss


def train_irtr(pl_module, batch):
    '''没有添加负样本队列
    '''
    with torch.no_grad():
        pl_module.temp.clamp_(0.1, 1.5)
    alpha = pl_module.hparams.config['cur_alpha']

    # 获得预训练模型输出的特征
    idx = batch['image_index'].view(-1, 1)
    image_feats, image_embeds, image_atts = pl_module.encoding_image(batch)
    text_feats, text_embeds, text_atts = pl_module.encoding_text(batch)

    sim_targets = torch.eye(len(image_feats), device=pl_module.device, dtype=torch.long)

    sim_i2t = image_feats @ text_feats.t()
    sim_t2i = text_feats @ image_feats.t()

    loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
    loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()
    loss_itc = (loss_i2t + loss_t2i) / 2

    ###============== Negative sampling ===================###
    bs = image_feats.size(0)
    idxs = concat_all_gather(idx, world_size=pl_module.trainer.world_size)
    if pl_module.negative_all_rank:  # 如果是分布式，从所有卡中抽取负样本
        # compute sample similarity
        with torch.no_grad():
            mask = torch.eq(idx, idxs.t())
            image_feats_world = concat_all_gather(image_feats, pl_module.trainer.world_size)
            text_feats_world = concat_all_gather(text_feats, pl_module.trainer.world_size)
            # sim_i2t = image_feats @ text_feats_world.t() / pl_module.temp
            # sim_t2i = text_feats @ image_feats_world.t() / pl_module.temp
            sim_i2t = image_feats @ text_feats_world.t()
            sim_t2i = text_feats @ image_feats_world.t()

            weights_i2t = F.softmax(sim_i2t, dim=1)
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_t2i.masked_fill_(mask, 0)

        image_embeds_world = all_gather_with_grad(image_embeds, pl_module.trainer.world_size)
        text_embeds_world = all_gather_with_grad(text_embeds, pl_module.trainer.world_size)
        text_attns_world = all_gather_with_grad(text_atts, pl_module.trainer.world_size)

        # select a negative image (from all ranks) for each text
        image_embeds_neg = []
        image_feats_neg = []  # for triplet loss
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
            image_feats_neg.append(image_feats_world[neg_idx])

        # select a negative text (from all ranks) for each image
        text_embeds_neg = []
        text_attns_neg = []
        text_feats_neg = []  # for triplet loss
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds_world[neg_idx])
            text_attns_neg.append(text_attns_world[neg_idx])
            text_feats_neg.append(text_feats_world[neg_idx])
    else:  # 仅从当前卡上的批次中抽取负样本
        with torch.no_grad():
            mask = torch.eq(idx, idx.t())

            # sim_i2t = image_feats @ text_feats.t() / pl_module.temp
            # sim_t2i = text_feats @ image_feats.t() / pl_module.temp

            sim_i2t = image_feats @ text_feats.t()
            sim_t2i = text_feats @ image_feats.t()

            weights_i2t = F.softmax(sim_i2t, dim=1)
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_t2i.masked_fill_(mask, 0)

        # select a negative image (from same rank) for each text
        image_embeds_neg = []
        image_feats_neg = []  # for triplet loss
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
            image_feats_neg.append(image_feats[neg_idx])

        # select a negative text (from same rank) for each image
        text_embeds_neg = []
        text_attns_neg = []
        text_feats_neg = []  # for triplet loss
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_attns_neg.append(text_atts[neg_idx])
            text_feats_neg.append(text_feats[neg_idx])

    image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
    image_feats_neg = torch.stack(image_feats_neg, dim=0)
    text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
    text_attns_neg = torch.stack(text_attns_neg, dim=0)
    text_feats_neg = torch.stack(text_feats_neg, dim=0)
    ###============== Image-text Matching ===================###
    # 总共三对图文向量：正正、正负、负正
    image_hidden_states = torch.cat([image_embeds, image_embeds, image_embeds_neg], dim=0)
    image_attention_mask = torch.cat([image_atts, image_atts, image_atts], dim=0)
    text_hidden_states = torch.cat([text_embeds, text_embeds_neg, text_embeds], dim=0)
    text_attention_mask = torch.cat([text_atts, text_attns_neg, text_atts])

    # # 第0个位置表示正确的置信度，第1个位置表示错误的置信度
    # itm_labels = torch.cat([torch.zeros(bs, dtype=torch.long), torch.ones(2 * bs, dtype=torch.long)],dim=0).to(image_feats.device)
    itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],dim=0).to(image_feats.device)

    output_t2i = pl_module.aformer(
        text_hidden_states,
        attention_mask=text_attention_mask,
        encoder_hidden_states=image_hidden_states,
        encoder_attention_mask=image_attention_mask,
        mode='t2i'
    )

    output_i2t = pl_module.aformer(
        image_hidden_states,
        attention_mask=image_attention_mask,
        encoder_hidden_states=text_hidden_states,
        encoder_attention_mask=text_attention_mask,
        mode='i2t'
    )

    t2i_embedding = output_t2i.last_hidden_state[:, 0]
    i2t_embedding = output_i2t.last_hidden_state[:, 0]
    # t2i_embedding = output_t2i.pooler_output
    # i2t_embedding = output_i2t.pooler_output
    vl_t2i = pl_module.itm_head(t2i_embedding)
    vl_i2t = pl_module.itm_head(i2t_embedding)

    loss_itm = (F.cross_entropy(vl_t2i, itm_labels) + F.cross_entropy(vl_i2t, itm_labels)) / 2

    irtr_loss = loss_itm + loss_itc
    irtr_loss_ = getattr(pl_module, f"train_irtr_loss")(irtr_loss)
    pl_module.log(f"train/itc_loss", loss_itc)
    pl_module.log(f"train/itm_loss", loss_itm)
    # pl_module.log(f"train_irtr_loss/irtr/triplet_loss", loss_triplet)
    pl_module.log(f"train/total_loss", irtr_loss)

    # pl_module.log(f"t/loss_i2t", loss_i2t)
    # pl_module.log(f"t/loss_t2i", loss_t2i)
    # pl_module.log(f"t/loss_i2i_IM_g2l", loss_i2i_IM_g2l)
    # pl_module.log(f"t/loss_t2t_IM_g2l", loss_t2t_IM_g2l)
    # pl_module.log(f"t/loss_t2t", loss_t2t)
    # pl_module.log(f"t/loss_i2i", loss_i2i)
    return irtr_loss

def train_irtr_with_queue(pl_module, batch):
    '''
    添加负样本队列
    '''
    with torch.no_grad():
        pl_module.temp.clamp_(0.1, 3.0)
        pl_module.alpha.clamp_(0.2, 0.8)

    cur_alpha = pl_module.hparams.config['cur_alpha']
    # 获得预训练模型输出的特征
    idx = batch['image_index'].view(-1, 1)
    idx_all = torch.cat([idx.t(), pl_module.idx_queue.clone().detach()], dim=1)
    pos_idx = torch.eq(idx, idx_all).float()
    sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

    image_feats, image_embeds, image_atts = pl_module.encoding_image(batch)
    text_feats, text_embeds, text_atts = pl_module.encoding_text(batch)

    with torch.no_grad():
        image_feats_q = image_feats.clone().detach()
        text_feats_q = text_feats.clone().detach()
        image_feat_all = torch.cat([image_feats_q.t(), pl_module.image_queue.clone().detach()], dim=1)
        text_feat_all = torch.cat([text_feats_q.t(), pl_module.text_queue.clone().detach()], dim=1)

    sim_i2t = image_feats @ text_feat_all / pl_module.temp
    sim_t2i = text_feats @ image_feat_all / pl_module.temp
    # sim_i2t = image_feats @ text_feat_all
    # sim_t2i = text_feats @ image_feat_all

    if pl_module.distill:
        sim_i2t_q  = image_feats @ text_feat_all / pl_module.temp
        sim_t2i_q  = text_feats @ image_feat_all / pl_module.temp
        # sim_i2t_q = image_feats @ text_feat_all
        # sim_t2i_q = text_feats @ image_feat_all

        sim_i2t_targets = cur_alpha * F.softmax(sim_i2t_q, dim=1) + (1 - cur_alpha) * sim_targets
        sim_t2i_targets = cur_alpha * F.softmax(sim_t2i_q, dim=1) + (1 - cur_alpha) * sim_targets

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

    else:
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean()

    loss_itc = (loss_i2t + loss_t2i) / 2
    pl_module._dequeue_and_enqueue(image_feats_q, text_feats_q, idx)

    ###============== Negative sampling ===================###
    bs = image_feats.size(0)
    idxs = concat_all_gather(idx, world_size=pl_module.trainer.world_size)
    if pl_module.negative_all_rank:  # 如果是分布式，从所有卡中抽取负样本
        # compute sample similarity
        with torch.no_grad():
            mask = torch.eq(idx, idxs.t())
            image_feats_world = concat_all_gather(image_feats, pl_module.trainer.world_size)
            text_feats_world = concat_all_gather(text_feats, pl_module.trainer.world_size)

            sim_i2t = image_feats @ text_feats_world.t() / pl_module.temp
            sim_t2i = text_feats @ image_feats_world.t() / pl_module.temp
            # sim_i2t = image_feats @ text_feats_world.t()
            # sim_t2i = text_feats @ image_feats_world.t()

            weights_i2t = F.softmax(sim_i2t, dim=1)
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_t2i.masked_fill_(mask, 0)

        image_embeds_world = all_gather_with_grad(image_embeds, pl_module.trainer.world_size)
        text_embeds_world = all_gather_with_grad(text_embeds, pl_module.trainer.world_size)
        text_attns_world = all_gather_with_grad(text_atts, pl_module.trainer.world_size)

        # select a negative image (from all ranks) for each text
        image_embeds_neg = []
        image_feats_neg = []  # for triplet loss
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
            image_feats_neg.append(image_feats_world[neg_idx])

        # select a negative text (from all ranks) for each image
        text_embeds_neg = []
        text_attns_neg = []
        text_feats_neg = []  # for triplet loss
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds_world[neg_idx])
            text_attns_neg.append(text_attns_world[neg_idx])
            text_feats_neg.append(text_feats_world[neg_idx])
    else:  # 仅从当前卡上的批次中抽取负样本
        with torch.no_grad():
            mask = torch.eq(idx, idx.t())

            sim_i2t = image_feats @ text_feats.t() / pl_module.temp
            sim_t2i = text_feats @ image_feats.t() / pl_module.temp

            # sim_i2t = image_feats @ text_feats.t()
            # sim_t2i = text_feats @ image_feats.t()

            weights_i2t = F.softmax(sim_i2t, dim=1)
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_t2i.masked_fill_(mask, 0)

        # select a negative image (from same rank) for each text
        image_embeds_neg = []
        image_feats_neg = []  # for triplet loss
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
            image_feats_neg.append(image_feats[neg_idx])

        # select a negative text (from same rank) for each image
        text_embeds_neg = []
        text_attns_neg = []
        text_feats_neg = []  # for triplet loss
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_attns_neg.append(text_atts[neg_idx])
            text_feats_neg.append(text_feats[neg_idx])

    image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
    image_feats_neg = torch.stack(image_feats_neg, dim=0)
    text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
    text_attns_neg = torch.stack(text_attns_neg, dim=0)
    text_feats_neg = torch.stack(text_feats_neg, dim=0)
    ###============== Image-text Matching ===================###
    # 总共三对图文向量：正正、正负、负正
    image_hidden_states = torch.cat([image_embeds, image_embeds, image_embeds_neg], dim=0)
    image_attention_mask = torch.cat([image_atts, image_atts, image_atts], dim=0)
    text_hidden_states = torch.cat([text_embeds, text_embeds_neg, text_embeds], dim=0)
    text_attention_mask = torch.cat([text_atts, text_attns_neg, text_atts])

    # # 第0个位置表示正确的置信度，第1个位置表示错误的置信度
    # itm_labels = torch.cat([torch.zeros(bs, dtype=torch.long), torch.ones(2 * bs, dtype=torch.long)],dim=0).to(image_feats.device)
    itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],dim=0).to(image_feats.device)

    output_t2i = pl_module.aformer(
        text_hidden_states,
        attention_mask=text_attention_mask,
        encoder_hidden_states=image_hidden_states,
        encoder_attention_mask=image_attention_mask,
        mode='t2i'
    )

    output_i2t = pl_module.aformer(
        image_hidden_states,
        attention_mask=image_attention_mask,
        encoder_hidden_states=text_hidden_states,
        encoder_attention_mask=text_attention_mask,
        mode='i2t'
    )

    t2i_embedding = output_t2i.last_hidden_state[:, 0]
    i2t_embedding = output_i2t.last_hidden_state[:, 0]
    # t2i_embedding = output_t2i.pooler_output
    # i2t_embedding = output_i2t.pooler_output
    vl_t2i = pl_module.itm_head(t2i_embedding)
    vl_i2t = pl_module.itm_head(i2t_embedding)

    loss_itm = (F.cross_entropy(vl_t2i, itm_labels) + F.cross_entropy(vl_i2t, itm_labels)) / 2

    irtr_loss = loss_itm + loss_itc
    irtr_loss_ = getattr(pl_module, f"train_irtr_loss")(irtr_loss)
    pl_module.log(f"train/itc_loss", loss_itc)
    pl_module.log(f"train/itm_loss", loss_itm)
    # pl_module.log(f"train_irtr_loss/irtr/triplet_loss", loss_triplet)
    pl_module.log(f"train/total_loss", irtr_loss)
    pl_module.log(f"train/temp", pl_module.temp)
    pl_module.log(f"train/alpha", pl_module.alpha)




    # pl_module.log(f"t/loss_i2t", loss_i2t)
    # pl_module.log(f"t/loss_t2i", loss_t2i)
    # pl_module.log(f"t/loss_i2i_IM_g2l", loss_i2i_IM_g2l)
    # pl_module.log(f"t/loss_t2t_IM_g2l", loss_t2t_IM_g2l)
    # pl_module.log(f"t/loss_t2t", loss_t2t)
    # pl_module.log(f"t/loss_i2i", loss_i2i)
    return irtr_loss

def train_irtr_with_double_queue(pl_module, batch):
    '''
    添加全局和局部的负样本队列
    '''
    with torch.no_grad():
        pl_module.temp.clamp_(0.1, 3.0)
        pl_module.alpha.clamp_(0.2, 0.8)

    cur_alpha = pl_module.hparams.config['cur_alpha']
    # 获得预训练模型输出的特征
    idx = batch['image_index'].view(-1, 1)
    idx_all = torch.cat([idx.t(), pl_module.idx_queue.clone().detach()], dim=1)
    pos_idx = torch.eq(idx, idx_all).float()
    sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

    image_feats, image_embeds, image_atts = pl_module.encoding_image(batch)
    text_feats, text_embeds, text_atts = pl_module.encoding_text(batch)

    with torch.no_grad():
        image_feats_q = image_feats.clone().detach()
        text_feats_q = text_feats.clone().detach()
        image_embeds_q = image_embeds.clone().detach()
        text_embeds_q = text_embeds.clone().detach()
        # image_atts_q = image_atts.clone().detach()
        text_atts_q = text_atts.clone().detach()

        image_feat_all = torch.cat([image_feats_q.t(), pl_module.image_queue.clone().detach()], dim=1)
        text_feat_all = torch.cat([text_feats_q.t(), pl_module.text_queue.clone().detach()], dim=1)
        image_embeds_all = torch.cat([image_embeds_q, pl_module.image_embed_queue.clone().detach()], dim=0)
        text_embeds_all = torch.cat([text_embeds_q, pl_module.text_embed_queue.clone().detach()], dim=0)
        # image_attn_all = torch.cat([image_atts_q, pl_module.image_attn_queue.clone().detach()], dim=0)
        text_attn_all = torch.cat([text_atts_q, pl_module.text_attn_queue.clone().detach()], dim=0)

    sim_i2t = image_feats @ text_feat_all / pl_module.temp
    sim_t2i = text_feats @ image_feat_all / pl_module.temp

    if pl_module.distill:
        sim_i2t_q  = image_feats @ text_feat_all / pl_module.temp
        sim_t2i_q  = text_feats @ image_feat_all / pl_module.temp
        # sim_i2t_q = image_feats @ text_feat_all
        # sim_t2i_q = text_feats @ image_feat_all

        sim_i2t_targets = cur_alpha * F.softmax(sim_i2t_q, dim=1) + (1 - cur_alpha) * sim_targets
        sim_t2i_targets = cur_alpha * F.softmax(sim_t2i_q, dim=1) + (1 - cur_alpha) * sim_targets

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

    else:
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean()

    loss_itc = (loss_i2t + loss_t2i) / 2
    pl_module._dequeue_and_enqueue_double(
        image_feats_q, text_feats_q, image_embeds_q, text_embeds_q, text_atts_q, idx)

    ###============== Negative sampling ===================###
    bs = image_feats.size(0)
    # idxs = concat_all_gather(idx, world_size=pl_module.trainer.world_size)
    mask = torch.eq(idx, idx_all)
    weights_i2t = F.softmax(sim_i2t, dim=1)
    weights_i2t.masked_fill_(mask, 0)
    weights_t2i = F.softmax(sim_t2i, dim=1)
    weights_t2i.masked_fill_(mask, 0)

    # select a negative image (from all ranks) for each text
    image_embeds_neg = []
    image_feats_neg = []  # for triplet loss
    for b in range(bs):
        neg_idx = torch.multinomial(weights_t2i[b], 1).item()
        image_embeds_neg.append(image_embeds_all[neg_idx])
        image_feats_neg.append(image_feat_all[neg_idx])

    # select a negative text (from all ranks) for each image
    text_embeds_neg = []
    text_attns_neg = []
    text_feats_neg = []  # for triplet loss
    for b in range(bs):
        neg_idx = torch.multinomial(weights_i2t[b], 1).item()
        text_embeds_neg.append(text_embeds_all[neg_idx])
        text_attns_neg.append(text_attn_all[neg_idx])
        text_feats_neg.append(text_feat_all[neg_idx])

    # if pl_module.negative_all_rank:  # 如果是分布式，从所有卡中抽取负样本
    #     # compute sample similarity
    #     with torch.no_grad():
    #         mask = torch.eq(idx, idxs.t())
    #         image_feats_world = concat_all_gather(image_feats, pl_module.trainer.world_size)
    #         text_feats_world = concat_all_gather(text_feats, pl_module.trainer.world_size)
    #
    #         sim_i2t = image_feats @ text_feats_world.t() / pl_module.temp
    #         sim_t2i = text_feats @ image_feats_world.t() / pl_module.temp
    #         # sim_i2t = image_feats @ text_feats_world.t()
    #         # sim_t2i = text_feats @ image_feats_world.t()
    #
    #         weights_i2t = F.softmax(sim_i2t, dim=1)
    #         weights_i2t.masked_fill_(mask, 0)
    #         weights_t2i = F.softmax(sim_t2i, dim=1)
    #         weights_t2i.masked_fill_(mask, 0)
    #
    #     image_embeds_world = all_gather_with_grad(image_embeds, pl_module.trainer.world_size)
    #     text_embeds_world = all_gather_with_grad(text_embeds, pl_module.trainer.world_size)
    #     text_attns_world = all_gather_with_grad(text_atts, pl_module.trainer.world_size)
    #
    #     # select a negative image (from all ranks) for each text
    #     image_embeds_neg = []
    #     image_feats_neg = []  # for triplet loss
    #     for b in range(bs):
    #         neg_idx = torch.multinomial(weights_t2i[b], 1).item()
    #         image_embeds_neg.append(image_embeds_world[neg_idx])
    #         image_feats_neg.append(image_feats_world[neg_idx])
    #
    #     # select a negative text (from all ranks) for each image
    #     text_embeds_neg = []
    #     text_attns_neg = []
    #     text_feats_neg = []  # for triplet loss
    #     for b in range(bs):
    #         neg_idx = torch.multinomial(weights_i2t[b], 1).item()
    #         text_embeds_neg.append(text_embeds_world[neg_idx])
    #         text_attns_neg.append(text_attns_world[neg_idx])
    #         text_feats_neg.append(text_feats_world[neg_idx])
    # else:  # 仅从当前卡上的批次中抽取负样本
    #     with torch.no_grad():
    #         mask = torch.eq(idx, idx.t())
    #
    #         sim_i2t = image_feats @ text_feats.t() / pl_module.temp
    #         sim_t2i = text_feats @ image_feats.t() / pl_module.temp
    #
    #         # sim_i2t = image_feats @ text_feats.t()
    #         # sim_t2i = text_feats @ image_feats.t()
    #
    #         weights_i2t = F.softmax(sim_i2t, dim=1)
    #         weights_i2t.masked_fill_(mask, 0)
    #         weights_t2i = F.softmax(sim_t2i, dim=1)
    #         weights_t2i.masked_fill_(mask, 0)
    #
    #     # select a negative image (from same rank) for each text
    #     image_embeds_neg = []
    #     image_feats_neg = []  # for triplet loss
    #     for b in range(bs):
    #         neg_idx = torch.multinomial(weights_t2i[b], 1).item()
    #         image_embeds_neg.append(image_embeds[neg_idx])
    #         image_feats_neg.append(image_feats[neg_idx])
    #
    #     # select a negative text (from same rank) for each image
    #     text_embeds_neg = []
    #     text_attns_neg = []
    #     text_feats_neg = []  # for triplet loss
    #     for b in range(bs):
    #         neg_idx = torch.multinomial(weights_i2t[b], 1).item()
    #         text_embeds_neg.append(text_embeds[neg_idx])
    #         text_attns_neg.append(text_atts[neg_idx])
    #         text_feats_neg.append(text_feats[neg_idx])

    image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
    image_feats_neg = torch.stack(image_feats_neg, dim=0)
    text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
    text_attns_neg = torch.stack(text_attns_neg, dim=0)
    text_feats_neg = torch.stack(text_feats_neg, dim=0)
    ###============== Image-text Matching ===================###
    # 总共三对图文向量：正正、正负、负正
    image_hidden_states = torch.cat([image_embeds, image_embeds, image_embeds_neg], dim=0)
    image_attention_mask = torch.cat([image_atts, image_atts, image_atts], dim=0)
    text_hidden_states = torch.cat([text_embeds, text_embeds_neg, text_embeds], dim=0)
    text_attention_mask = torch.cat([text_atts, text_attns_neg, text_atts])

    # # 第0个位置表示正确的置信度，第1个位置表示错误的置信度
    # itm_labels = torch.cat([torch.zeros(bs, dtype=torch.long), torch.ones(2 * bs, dtype=torch.long)],dim=0).to(image_feats.device)
    itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],dim=0).to(image_feats.device)

    output_t2i = pl_module.aformer(
        text_hidden_states,
        attention_mask=text_attention_mask,
        encoder_hidden_states=image_hidden_states,
        encoder_attention_mask=image_attention_mask,
        mode='t2i'
    )

    output_i2t = pl_module.aformer(
        image_hidden_states,
        attention_mask=image_attention_mask,
        encoder_hidden_states=text_hidden_states,
        encoder_attention_mask=text_attention_mask,
        mode='i2t'
    )

    t2i_embedding = output_t2i.last_hidden_state[:, 0]
    i2t_embedding = output_i2t.last_hidden_state[:, 0]
    # t2i_embedding = output_t2i.pooler_output
    # i2t_embedding = output_i2t.pooler_output
    vl_t2i = pl_module.itm_head(t2i_embedding)
    vl_i2t = pl_module.itm_head(i2t_embedding)

    loss_itm = (F.cross_entropy(vl_t2i, itm_labels) + F.cross_entropy(vl_i2t, itm_labels)) / 2

    irtr_loss = loss_itm + loss_itc
    irtr_loss_ = getattr(pl_module, f"train_irtr_loss")(irtr_loss)
    pl_module.log(f"train/itc_loss", loss_itc)
    pl_module.log(f"train/itm_loss", loss_itm)
    # pl_module.log(f"train_irtr_loss/irtr/triplet_loss", loss_triplet)
    pl_module.log(f"train/total_loss", irtr_loss)
    pl_module.log(f"train/temp", pl_module.temp)
    pl_module.log(f"train/alpha", pl_module.alpha)




    # pl_module.log(f"t/loss_i2t", loss_i2t)
    # pl_module.log(f"t/loss_t2i", loss_t2i)
    # pl_module.log(f"t/loss_i2i_IM_g2l", loss_i2i_IM_g2l)
    # pl_module.log(f"t/loss_t2t_IM_g2l", loss_t2t_IM_g2l)
    # pl_module.log(f"t/loss_t2t", loss_t2t)
    # pl_module.log(f"t/loss_i2i", loss_i2i)
    return irtr_loss

def train_irtr_with_queue_multi_out(pl_module, batch):
    '''
    添加负样本队列
    '''
    with torch.no_grad():
        pl_module.temp.clamp_(0.1, 3.0)
        pl_module.alpha.clamp_(0.2, 0.8)

    cur_alpha = pl_module.hparams.config['cur_alpha']
    # 获得预训练模型输出的特征
    idx = batch['image_index'].view(-1, 1)
    idx_all = torch.cat([idx.t(), pl_module.idx_queue.clone().detach()], dim=1)
    pos_idx = torch.eq(idx, idx_all).float()
    sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

    image_feats, image_embeds, image_atts = pl_module.encoding_image(batch)
    text_feats, text_embeds, text_atts = pl_module.encoding_text(batch)

    with torch.no_grad():
        image_feats_q = image_feats.clone().detach()
        text_feats_q = text_feats.clone().detach()
        image_feat_all = torch.cat([image_feats_q.t(), pl_module.image_queue.clone().detach()], dim=1)
        text_feat_all = torch.cat([text_feats_q.t(), pl_module.text_queue.clone().detach()], dim=1)

    sim_i2t = image_feats @ text_feat_all / pl_module.temp
    sim_t2i = text_feats @ image_feat_all / pl_module.temp
    # sim_i2t = image_feats @ text_feat_all
    # sim_t2i = text_feats @ image_feat_all

    if pl_module.distill:
        sim_i2t_q  = image_feats @ text_feat_all / pl_module.temp
        sim_t2i_q  = text_feats @ image_feat_all / pl_module.temp
        # sim_i2t_q = image_feats @ text_feat_all
        # sim_t2i_q = text_feats @ image_feat_all

        sim_i2t_targets = cur_alpha * F.softmax(sim_i2t_q, dim=1) + (1 - cur_alpha) * sim_targets
        sim_t2i_targets = cur_alpha * F.softmax(sim_t2i_q, dim=1) + (1 - cur_alpha) * sim_targets

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

    else:
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean()

    loss_itc = (loss_i2t + loss_t2i) / 2
    pl_module._dequeue_and_enqueue(image_feats_q, text_feats_q, idx)

    ###============== Negative sampling ===================###
    bs = image_feats.size(0)
    idxs = concat_all_gather(idx, world_size=pl_module.trainer.world_size)
    if pl_module.negative_all_rank:  # 如果是分布式，从所有卡中抽取负样本
        # compute sample similarity
        with torch.no_grad():
            mask = torch.eq(idx, idxs.t())
            image_feats_world = concat_all_gather(image_feats, pl_module.trainer.world_size)
            text_feats_world = concat_all_gather(text_feats, pl_module.trainer.world_size)

            sim_i2t = image_feats @ text_feats_world.t() / pl_module.temp
            sim_t2i = text_feats @ image_feats_world.t() / pl_module.temp
            # sim_i2t = image_feats @ text_feats_world.t()
            # sim_t2i = text_feats @ image_feats_world.t()

            weights_i2t = F.softmax(sim_i2t, dim=1)
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_t2i.masked_fill_(mask, 0)

        image_embeds_world = all_gather_with_grad(image_embeds, pl_module.trainer.world_size)
        text_embeds_world = all_gather_with_grad(text_embeds, pl_module.trainer.world_size)
        text_attns_world = all_gather_with_grad(text_atts, pl_module.trainer.world_size)

        # select a negative image (from all ranks) for each text
        image_embeds_neg = []
        image_feats_neg = []  # for triplet loss
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
            image_feats_neg.append(image_feats_world[neg_idx])

        # select a negative text (from all ranks) for each image
        text_embeds_neg = []
        text_attns_neg = []
        text_feats_neg = []  # for triplet loss
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds_world[neg_idx])
            text_attns_neg.append(text_attns_world[neg_idx])
            text_feats_neg.append(text_feats_world[neg_idx])
    else:  # 仅从当前卡上的批次中抽取负样本
        with torch.no_grad():
            mask = torch.eq(idx, idx.t())

            sim_i2t = image_feats @ text_feats.t() / pl_module.temp
            sim_t2i = text_feats @ image_feats.t() / pl_module.temp

            # sim_i2t = image_feats @ text_feats.t()
            # sim_t2i = text_feats @ image_feats.t()

            weights_i2t = F.softmax(sim_i2t, dim=1)
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_t2i.masked_fill_(mask, 0)

        # select a negative image (from same rank) for each text
        image_embeds_neg = []
        image_feats_neg = []  # for triplet loss
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
            image_feats_neg.append(image_feats[neg_idx])

        # select a negative text (from same rank) for each image
        text_embeds_neg = []
        text_attns_neg = []
        text_feats_neg = []  # for triplet loss
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_attns_neg.append(text_atts[neg_idx])
            text_feats_neg.append(text_feats[neg_idx])

    image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
    image_feats_neg = torch.stack(image_feats_neg, dim=0)
    text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
    text_attns_neg = torch.stack(text_attns_neg, dim=0)
    text_feats_neg = torch.stack(text_feats_neg, dim=0)
    ###============== Image-text Matching ===================###
    # 总共三对图文向量：正正、正负、负正
    image_hidden_states = torch.cat([image_embeds, image_embeds, image_embeds_neg], dim=0)
    image_attention_mask = torch.cat([image_atts, image_atts, image_atts], dim=0)
    text_hidden_states = torch.cat([text_embeds, text_embeds_neg, text_embeds], dim=0)
    text_attention_mask = torch.cat([text_atts, text_attns_neg, text_atts])

    # # 第0个位置表示正确的置信度，第1个位置表示错误的置信度
    # itm_labels = torch.cat([torch.zeros(bs, dtype=torch.long), torch.ones(2 * bs, dtype=torch.long)],dim=0).to(image_feats.device)
    itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],dim=0).to(image_feats.device)
    itm_labels = itm_labels.repeat(2)
    output_t2i = pl_module.aformer(
        text_hidden_states,
        attention_mask=text_attention_mask,
        encoder_hidden_states=image_hidden_states,
        encoder_attention_mask=image_attention_mask,
        mode='t2i',
        output_hidden_states=True,
    )

    output_i2t = pl_module.aformer(
        image_hidden_states,
        attention_mask=image_attention_mask,
        encoder_hidden_states=text_hidden_states,
        encoder_attention_mask=text_attention_mask,
        mode='i2t',
        output_hidden_states=True,
    )

    t2i_embedding = torch.cat([output_t2i.hidden_states[-3][:,0], output_t2i.hidden_states[-1][:,0]])   # 96,768
    i2t_embedding = torch.cat([output_i2t.hidden_states[-3][:,0], output_i2t.hidden_states[-1][:,0]])

    # t2i_embedding = output_t2i.last_hidden_state[:, 0]
    # i2t_embedding = output_i2t.last_hidden_state[:, 0]
    # t2i_embedding = output_t2i.pooler_output
    # i2t_embedding = output_i2t.pooler_output
    vl_t2i = pl_module.itm_head(t2i_embedding)
    vl_i2t = pl_module.itm_head(i2t_embedding)

    loss_itm = (F.cross_entropy(vl_t2i, itm_labels) + F.cross_entropy(vl_i2t, itm_labels)) / 2

    irtr_loss = loss_itm + loss_itc
    irtr_loss_ = getattr(pl_module, f"train_irtr_loss")(irtr_loss)
    pl_module.log(f"train/itc_loss", loss_itc)
    pl_module.log(f"train/itm_loss", loss_itm)
    # pl_module.log(f"train_irtr_loss/irtr/triplet_loss", loss_triplet)
    pl_module.log(f"train/total_loss", irtr_loss)
    pl_module.log(f"train/temp", pl_module.temp)
    pl_module.log(f"train/alpha", pl_module.alpha)




    # pl_module.log(f"t/loss_i2t", loss_i2t)
    # pl_module.log(f"t/loss_t2i", loss_t2i)
    # pl_module.log(f"t/loss_i2i_IM_g2l", loss_i2i_IM_g2l)
    # pl_module.log(f"t/loss_t2t_IM_g2l", loss_t2t_IM_g2l)
    # pl_module.log(f"t/loss_t2t", loss_t2t)
    # pl_module.log(f"t/loss_i2i", loss_i2i)
    return irtr_loss