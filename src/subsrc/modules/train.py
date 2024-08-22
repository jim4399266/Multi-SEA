
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import functools
from collections import defaultdict
import torch.distributed as dist
from .dist_utils import all_gather_with_grad, concat_all_gather


def train_irtr_with_queue_multi_out(pl_module, batch):
    with torch.no_grad():
        pl_module.temp.clamp_(0.1, 3.0)
        pl_module.alpha.clamp_(0.2, 0.8)

    cur_alpha = pl_module.hparams.config['cur_alpha']

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


    if pl_module.distill:
        sim_i2t_q = image_feats @ text_feat_all / pl_module.temp
        sim_t2i_q = text_feats @ image_feat_all / pl_module.temp


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
    if pl_module.negative_all_rank:
        # compute sample similarity
        with torch.no_grad():
            mask = torch.eq(idx, idxs.t())
            image_feats_world = concat_all_gather(image_feats, pl_module.trainer.world_size)
            text_feats_world = concat_all_gather(text_feats, pl_module.trainer.world_size)

            sim_i2t = image_feats @ text_feats_world.t() / pl_module.temp
            sim_t2i = text_feats @ image_feats_world.t() / pl_module.temp

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
    else:
        with torch.no_grad():
            mask = torch.eq(idx, idx.t())

            sim_i2t = image_feats @ text_feats.t() / pl_module.temp
            sim_t2i = text_feats @ image_feats.t() / pl_module.temp


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
    # 3 pairs of image-text features: pos-pos, pos-neg, neg-pos
    image_hidden_states = torch.cat([image_embeds, image_embeds, image_embeds_neg], dim=0)
    image_attention_mask = torch.cat([image_atts, image_atts, image_atts], dim=0)
    text_hidden_states = torch.cat([text_embeds, text_embeds_neg, text_embeds], dim=0)
    text_attention_mask = torch.cat([text_atts, text_attns_neg, text_atts])


    itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],dim=0).to(image_feats.device)

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


    itm_labels = itm_labels.repeat(2)

    t2i_embedding = torch.cat([output_t2i.hidden_states[-3][:,0], output_t2i.hidden_states[-1][:,0]])   # 96,768
    i2t_embedding = torch.cat([output_i2t.hidden_states[-3][:,0], output_i2t.hidden_states[-1][:,0]])

    vl_t2i = pl_module.itm_head(t2i_embedding)
    vl_i2t = pl_module.itm_head(i2t_embedding)

    loss_itm = (F.cross_entropy(vl_t2i, itm_labels) + F.cross_entropy(vl_i2t, itm_labels)) / 2

    irtr_loss = loss_itm + loss_itc
    irtr_loss_ = getattr(pl_module, f"train_irtr_loss")(irtr_loss)
    pl_module.log(f"train/itc_loss", loss_itc)
    pl_module.log(f"train/itm_loss", loss_itm)

    pl_module.log(f"train/total_loss", irtr_loss)
    pl_module.log(f"train/temp", pl_module.temp)
    pl_module.log(f"train/alpha", pl_module.alpha)

    return irtr_loss