# import torch
# def process_en(word_en, targets):
#     positive_map = targets['positive_map'].sum(1)   # main object
#     modify_positive_map = targets["modify_positive_map"].sum(1)    # attribute(modify)
#     pron_positive_map = targets["pron_positive_map"].sum(1)        # pron
#     rel_positive_map = targets["rel_positive_map"].sum(1)        # relation  [B,num_objects,L]
#     pos_mask = positive_map + modify_positive_map + pron_positive_map + rel_positive_map
#     pos_mask = torch.where(pos_mask > 1, torch.tensor(1., dtype=pos_mask.dtype), pos_mask)
#     # 加一个prompt token
#     # pos_mask_prompt = torch.zeros(pos_mask.size(0), 1, device=pos_mask.device, dtype=pos_mask.dtype)
#     # pos_mask = torch.cat([pos_mask[:,:1], pos_mask_prompt, pos_mask[:,1:]], dim=1)
#     process_en_feature = word_en * pos_mask.unsqueeze(-1)


#     # return process_en_feature
#     return process_en_feature, pos_mask

# # def process_en(word_en, targets):
# #     positive_map = targets['positive_map'].sum(1)   # main object
# #     positive_map = torch.where(positive_map > 1, torch.tensor(1., dtype=positive_map.dtype), positive_map) #确保一下mask的0,1
# #     modify_positive_map = targets["modify_positive_map"].sum(1)    # attribute(modify)
# #     modify_positive_map = torch.where(modify_positive_map > 1, torch.tensor(1., dtype=modify_positive_map.dtype), modify_positive_map) 
# #     # pron_positive_map = targets["pron_positive_map"].sum(1)        # pron
# #     # pron_positive_map
# #     rel_positive_map = targets["rel_positive_map"].sum(1)        # relation  [B,num_objects,L]
# #     rel_positive_map = torch.where(rel_positive_map > 1, torch.tensor(1., dtype=rel_positive_map.dtype), rel_positive_map) 
# #     # pos_mask = positive_map + modify_positive_map + pron_positive_map + rel_positive_map
# #     # pos_mask = torch.where(pos_mask > 1, torch.tensor(1., dtype=pos_mask.dtype), pos_mask)
# #     # 加一个prompt token
# #     # pos_mask_prompt = torch.zeros(pos_mask.size(0), 1, device=pos_mask.device, dtype=pos_mask.dtype)
# #     # pos_mask = torch.cat([pos_mask[:,:1], pos_mask_prompt, pos_mask[:,1:]], dim=1)
# #     positive_mask_sum = positive_map.sum(dim=1, keepdim=True) + 1e-6
# #     has_posi = (positive_mask_sum > 1e-5).float()  # [B, 1]
# #     T_posi = ((word_en * positive_map.unsqueeze(-1)).sum(dim=1) / positive_mask_sum) * has_posi


# #     modify_mask_sum = modify_positive_map.sum(dim=1, keepdim=True) + 1e-6
# #     has_modify = (modify_mask_sum > 1e-5).float()  # [B, 1]
# #     Tmodi = ((word_en * modify_positive_map.unsqueeze(-1)).sum(dim=1) / modify_mask_sum) * has_modify  
    
# #     rel_mask_sum = rel_positive_map.sum(dim=1, keepdim=True) + 1e-6
# #     has_rel = (rel_mask_sum > 1e-5).float()  # [B, 1]
# #     Trel = ((word_en * rel_positive_map.unsqueeze(-1)).sum(dim=1) / rel_mask_sum) * has_rel    

    


# #     # return process_en_feature
# #     return T_posi, Tmodi, Trel

import torch


def _align_token_valid_mask(token_valid_mask, pos_mask):
    """
    Align token_valid_mask to pos_mask shape.
    This is only used to avoid perturbing padding tokens.
    """
    if token_valid_mask is None:
        return torch.ones_like(pos_mask, dtype=torch.bool)

    token_valid_mask = token_valid_mask.to(device=pos_mask.device).bool()

    if token_valid_mask.shape == pos_mask.shape:
        return token_valid_mask

    # Handle possible prompt-token length mismatch:
    # pos_mask: [B, L+1], token_valid_mask: [B, L]
    if (
        token_valid_mask.dim() == 2
        and pos_mask.dim() == 2
        and token_valid_mask.size(0) == pos_mask.size(0)
        and token_valid_mask.size(1) + 1 == pos_mask.size(1)
    ):
        prompt_valid = torch.ones(
            token_valid_mask.size(0),
            1,
            device=pos_mask.device,
            dtype=torch.bool,
        )
        return torch.cat([prompt_valid, token_valid_mask], dim=1)

    # Handle the opposite case if needed:
    # pos_mask: [B, L], token_valid_mask: [B, L+1]
    if (
        token_valid_mask.dim() == 2
        and pos_mask.dim() == 2
        and token_valid_mask.size(0) == pos_mask.size(0)
        and token_valid_mask.size(1) == pos_mask.size(1) + 1
    ):
        return token_valid_mask[:, 1:]

    # Fallback: if shapes do not match, perturb all positions.
    return torch.ones_like(pos_mask, dtype=torch.bool)


def apply_parser_mask_noise(
    pos_mask,
    parser_noise_ratio=0.0,
    parser_noise_mode="flip",
    token_valid_mask=None,
):
    """
    Add artificial noise to parser-derived token masks.

    Args:
        pos_mask: [B, L], binary parser-derived mask.
        parser_noise_ratio: float, e.g., 0.1, 0.2, 0.3.
        parser_noise_mode:
            - "flip": randomly flip 0/1 on valid tokens.
            - "drop": randomly drop selected tokens from 1 to 0.
            - "add": randomly add selected tokens from 0 to 1.
        token_valid_mask: [B, L], True for non-padding tokens.

    Returns:
        noisy_pos_mask: [B, L]
    """
    if parser_noise_ratio is None or parser_noise_ratio <= 0:
        return pos_mask

    parser_noise_ratio = float(parser_noise_ratio)
    parser_noise_ratio = max(0.0, min(parser_noise_ratio, 1.0))

    noisy_pos_mask = pos_mask.clone()
    valid_mask = _align_token_valid_mask(token_valid_mask, pos_mask)

    if parser_noise_mode == "drop":
        candidate_mask = valid_mask & (noisy_pos_mask > 0)
    elif parser_noise_mode == "add":
        candidate_mask = valid_mask & (noisy_pos_mask <= 0)
    elif parser_noise_mode == "flip":
        candidate_mask = valid_mask
    else:
        raise ValueError(
            f"Unknown parser_noise_mode: {parser_noise_mode}. "
            f"Expected one of ['flip', 'drop', 'add']."
        )

    noise_mask = (
        torch.rand(noisy_pos_mask.shape, device=noisy_pos_mask.device)
        < parser_noise_ratio
    ) & candidate_mask

    if parser_noise_mode == "drop":
        noisy_pos_mask = torch.where(
            noise_mask,
            torch.zeros_like(noisy_pos_mask),
            noisy_pos_mask,
        )
    elif parser_noise_mode == "add":
        noisy_pos_mask = torch.where(
            noise_mask,
            torch.ones_like(noisy_pos_mask),
            noisy_pos_mask,
        )
    else:  # flip
        noisy_pos_mask = torch.where(
            noise_mask,
            1.0 - noisy_pos_mask,
            noisy_pos_mask,
        )

    return noisy_pos_mask.clamp(0.0, 1.0)


def process_en(
    word_en,
    targets,
    parser_noise_ratio=0.0,
    parser_noise_mode="flip",
    token_valid_mask=None,
):
    positive_map = targets["positive_map"].sum(1)                 # main object
    modify_positive_map = targets["modify_positive_map"].sum(1)   # attribute
    pron_positive_map = targets["pron_positive_map"].sum(1)       # pronoun
    rel_positive_map = targets["rel_positive_map"].sum(1)         # relation

    pos_mask = (
        positive_map
        + modify_positive_map
        + pron_positive_map
        + rel_positive_map
    )

    # Convert to binary mask. This is safer than torch.where(... torch.tensor(...)),
    # especially when pos_mask is on CUDA.
    pos_mask = (pos_mask > 0).to(dtype=word_en.dtype, device=word_en.device)

    # Add artificial parser noise for sensitivity analysis.
    pos_mask = apply_parser_mask_noise(
        pos_mask=pos_mask,
        parser_noise_ratio=parser_noise_ratio,
        parser_noise_mode=parser_noise_mode,
        token_valid_mask=token_valid_mask,
    )

    process_en_feature = word_en * pos_mask.unsqueeze(-1)

    return process_en_feature, pos_mask