import torch
import torch.nn.functional as F

def loss_pos_align(fi, targets, masks):
    fi = fi.permute(0,3,1,2)   #b,l,h,w
    if fi.shape[-2:] != masks.shape[-2:]:
        masks = F.interpolate(masks, fi.shape[-2:], mode='nearest').detach()
    fi = fi.permute(0,2,3,1)
    B, H, W, L = fi.shape
    fi = fi.reshape(B,H*W,L)
    logits = fi.log_softmax(-1)  
    for t in targets:
        continue
    # text position label
    positive_map = targets['positive_map'].sum(1)   
                   # main object
    modify_positive_map = targets["modify_positive_map"].sum(1)    # attribute(modify)
    pron_positive_map = targets["pron_positive_map"].sum(1)        # pron
    other_entity_map = targets["other_entity_map"].sum(1)          # other(auxi)
    rel_positive_map = targets["rel_positive_map"].sum(1)        # relation  [B,num_objects,L]
    #[batch,1,num_token]

    tgt_weight_pos = positive_map * 0.6 + modify_positive_map * 0.2 + pron_positive_map * 0.2 + rel_positive_map*0.1
    


    # mask, keep the positive term
    pos_mask = positive_map + modify_positive_map + pron_positive_map + rel_positive_map + other_entity_map

  
    masks = masks.permute(0,2,3,1)#b,h,w,1
    pos_mask_expanded = pos_mask.unsqueeze(1).unsqueeze(2) #b,1,1,L
    target_mask = masks * pos_mask_expanded #b,h,w,L
    b, h, w, L = target_mask.shape
    target_mask = target_mask.reshape(b,h*w,L)
    tgt_weight_pos_expanded = tgt_weight_pos.unsqueeze(1).unsqueeze(2)
    target_sim = masks * tgt_weight_pos_expanded
    target_sim = target_sim.reshape(b,h*w,L)
    
    

    # STEP Compute entropy
    entropy = torch.log(target_sim + 1e-6) * target_sim
    loss_ce = (entropy - logits * target_sim).sum(-1)
    loss_ce = loss_ce.mean()

    

    return loss_ce

def loss_sem_align(fi, targets, masks, word_en, pad_mask_en, temperature=0.07, eos_coef=0.1):
    if fi.shape[-2:] != masks.shape[-2:]:
        masks = F.interpolate(masks, fi.shape[-2:], mode='nearest').detach()
    
    masks = masks.permute(0,2,3,1)  #b,h,w,1
    B,H,W,C = masks.shape
    masks = masks.reshape(B,H*W,C) #b,h*w,1

    b,c,h,w = fi.shape
    fi = fi.permute(0,2,3,1)
    fi = fi.reshape(b,h*w,c)
    logits = (
        torch.matmul(fi, word_en.transpose(-1, -2))/temperature
        
    )  # [[B, h*w, num_tokens]
    #图像query特征与文本特征的余弦相似度
    # step 2. positive map
    # construct a map such that positive_map[k, i, j] = True
    # iff query i is associated to token j in batch item k

    # handle 'not mentioned'
    inds = ~pad_mask_en.sum(1) - 1  #获取每个样本文本的最后一个非pad token的位置
    # mask = torch.full(
    #     logits.shape[:2],
    #     eos_coef,
    #     dtype=torch.float32, device=logits.device
    #     )   # [B, L]
    # mask = torch.where(masks == 1, masks.float(), mask)
    positive_map = targets['positive_map'].sum(1)   #                 # main object
    positive_map[torch.arange(len(inds)), inds] = 0.5
    positive_map[torch.arange(len(inds)), inds-1] = 0.5
    
    positive_map_expanded = positive_map.unsqueeze(1) #b,1,l
    positive_map = masks * positive_map_expanded #b,h*w,L
    positive_map = positive_map > 0
    positive_logits = -logits.masked_fill(~positive_map, 0)
    negative_logits = logits

    modify_positive_map = targets["modify_positive_map"].sum(1)    # attribute(modify)
    modify_positive_map_expanded = modify_positive_map.unsqueeze(1) #b,1,l
    modify_positive_map =  masks * modify_positive_map_expanded #b,h*w,L
    modi_positive_logits = -logits.masked_fill(~(modify_positive_map>0), 0)
        
        
    pron_positive_map = targets["pron_positive_map"].sum(1)       # pron
    pron_positive_map_expanded = pron_positive_map.unsqueeze(1) #b,1,l
    pron_positive_map =  masks * pron_positive_map_expanded #b,h*w,L
    pron_positive_logits = -logits.masked_fill(~(pron_positive_map>0), 0)

    other_entity_map = targets["other_entity_map"].sum(1)         # other(auxi)
    other_entity_map_expanded = other_entity_map.unsqueeze(1) #b,1,l
    other_entity_map  =  masks * other_entity_map_expanded #b,h*w,L
    other_entity_neg_term = negative_logits.masked_fill(~(other_entity_map>0), 0)

    rel_positive_map = targets["rel_positive_map"].sum(1)        # relation  [B,num_objects,L]
    rel_positive_map_expanded = rel_positive_map.unsqueeze(1) #b,1,l
    rel_positive_map =  masks * rel_positive_map_expanded #b,h*w,L
    rel_positive_logits = -logits.masked_fill(~(rel_positive_map>0), 0)

   
    # step text mask
    # Token mask for matches <> 'not mentioned'

    pos_term = positive_logits.sum(2)
    pos_modi_term = modi_positive_logits.sum(2)
    pos_pron_term = pron_positive_logits.sum(2)
    pos_rel_term = rel_positive_logits.sum(2)
    # number of the token
    nb_modi_pos_token = (modify_positive_map>0).sum(2) + 1e-6
    nb_pron_pos_token = (pron_positive_map>0).sum(2) + 1e-6
    nb_rel_pos_token = (rel_positive_map>0).sum(2) + 1e-6

    ###############################
    # NOTE loss1: object --> text #
    ###############################
    pixels_with_pos = positive_map.any(2)   #对token取的，只要query对应的token里面有正样本，该值为True
    
    # note negative term
    neg_term = (negative_logits+other_entity_neg_term).logsumexp(2) 
    nb_pos_token = positive_map.sum(2) + 1e-6
    
    pixel_to_token_loss = (
       pos_term/nb_pos_token \
        + 0.2*pos_modi_term/nb_modi_pos_token \
        + 0.2*pos_pron_term /nb_pron_pos_token \
        + 0.1* pos_rel_term /nb_rel_pos_token \
        + neg_term
    ).masked_fill(~pixels_with_pos, 0)
    pixel_to_token_loss = pixel_to_token_loss.mean()
    return pixel_to_token_loss

 