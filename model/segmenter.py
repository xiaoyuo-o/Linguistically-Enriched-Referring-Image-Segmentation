import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model

from .layers import FPN, Projector, TransformerDecoder
from tools.losses import *
from tools.process_caption_parse import *


class SoftParserPrior(nn.Module):
    """Learnable token gate guided by, but not limited by, parser masks."""
    def __init__(self, word_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(word_dim * 2, word_dim),
            nn.GELU(),
            nn.Linear(word_dim, 1),
        )

    def forward(self, word, parser_mask):
        valid = (word.abs().sum(dim=-1, keepdim=True) > 0).to(word.dtype)
        sentence = (word * valid).sum(dim=1, keepdim=True) / (valid.sum(dim=1, keepdim=True) + 1e-6)
        sentence = sentence.expand(-1, word.size(1), -1)
        learned = torch.sigmoid(self.gate(torch.cat([word, sentence], dim=-1)))
        learned = 0.1 + 0.9 * learned
        parser_prior = 0.5 + 0.5 * parser_mask.to(word.dtype).unsqueeze(-1)
        return learned * parser_prior



class FeatureToTokenDistribution(nn.Module):
    def __init__(self, input_dim, token_len):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),  # 可选：加深网络
            nn.ReLU(),
            nn.Linear(input_dim, token_len)   # 最终输出为 token 序列长度 L
        )

    def forward(self, x):
        """
        input:
            x: [B, C, H, W] 
        output:
            out: [B, H, W, L] 
        """
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)          # [B, H, W, C]
        x = x.reshape(B * H * W, C)        # flatten  [B*H*W, C]
        out = self.mlp(x)                  # MLP [B*H*W, L]
        out = out.view(B, H, W, -1)        # reshape [B, H, W, L]
        return out
    

class CRIS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain, map_location="cuda:0")
        clip_model = clip_model.cuda().eval()
        # clip_model = torch.jit.load(cfg.clip_pretrain,
        #                             map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict()).float()
        # Multi-Modal FPN
        self.neck = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)
        # Decoder
        self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
                                          d_model=cfg.vis_dim,
                                          nhead=cfg.num_head,
                                          dim_ffn=cfg.dim_ffn,
                                          dropout=cfg.dropout,
                                          return_intermediate=cfg.intermediate)
        # Projector
        self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)
        self.vis_dim = cfg.vis_dim
        self.word_length = cfg.word_len
        self.image_projector = FeatureToTokenDistribution(input_dim=cfg.vis_dim, token_len=cfg.word_en_length)

        self.linear = nn.Linear(1024,cfg.word_dim)
        self.use_soft_parser_prior = bool(getattr(cfg, "soft_parser_prior", False))
        self.soft_parser_prior = SoftParserPrior(cfg.word_dim) if self.use_soft_parser_prior else None


        self.parser_noise_ratio = float(getattr(cfg, "parser_noise_ratio", 0.0))
        self.parser_noise_mode = getattr(cfg, "parser_noise_mode", "flip")

    def _apply_soft_parser_prior(self, word, feature, parser_mask):
        if not self.use_soft_parser_prior:
            return feature
        return word * self.soft_parser_prior(word, parser_mask)
    def forward(self, img, word, en_labels, en_labels_raw, mask=None, if_infer=False):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        if if_infer:
            text_raw = word
            # pad_mask_raw = torch.zeros_like(text_raw).masked_fill_(text_raw == 0, 1)  #b,60
            pad_mask_raw = torch.zeros_like(text_raw).masked_fill_(text_raw == 0, 1).bool() 
            vis = self.backbone.encode_image(img)
            word_raw, state_raw = self.backbone.encode_text(word,self.word_length,if_infer)  #此时 word_raw已经变成了max_length+1(多的是prompt token)，在第一个位置是prompt token
            # prompt_pad = torch.zeros(pad_mask_raw.size(0), 1, device=pad_mask_raw.device, dtype=pad_mask_raw.dtype)  #prompt token 的位置pad是0，表示该处不是pad token
            # pad_mask_raw = torch.cat([pad_mask_raw[:,:1], prompt_pad, pad_mask_raw[:,1:]], dim=1).bool()
            process_raw_feature = None
            process_raw_feature_mask = None
            if en_labels is not None:
                process_raw_feature, process_raw_feature_mask = process_en(
                    word_raw,
                    en_labels,
                    parser_noise_ratio=self.parser_noise_ratio,
                    parser_noise_mode=self.parser_noise_mode,
                    token_valid_mask=~pad_mask_raw,
                )                
                # process_raw_feature, process_raw_feature_mask = process_en(word_raw, en_labels)
                # pad_token = torch.ones((process_en_feature_mask.size(0), 1), dtype=process_en_feature_mask.dtype, device=process_en_feature_mask.device)

                    # 拼接在最后
                process_raw_feature_mask = process_raw_feature_mask.bool()
                process_raw_feature = self._apply_soft_parser_prior(
                    word_raw, process_raw_feature, process_raw_feature_mask)
                # process_en_feature_mask = torch.cat([process_en_feature_mask, pad_token], dim=1).bool()  # shape: [batch_size, 61]
                state_raw_ = state_raw.unsqueeze(1)
                process_raw_feature = torch.cat([process_raw_feature, state_raw_.expand(-1,process_raw_feature.shape[1],-1)],dim=-1)
                linear = self.linear.to(process_raw_feature.device)
                process_raw_feature = process_raw_feature.to(dtype=next(linear.parameters()).dtype)
                process_raw_feature = linear(process_raw_feature)


            if process_raw_feature is not None and process_raw_feature.dtype != word_raw.dtype:
                process_raw_feature = process_raw_feature.to(dtype=word_raw.dtype)
            # b, 512, 26, 26 (C4)
            fq_raw_ = self.neck(vis, state_raw)
            
            
            b, c, h, w = fq_raw_.size()
            #prompt token
            # style_logits = self.style_classifier(state_raw)
            # style_pred = torch.argmax(style_logits, dim=-1)            
            # style_embed = self.style_embeddings(style_pred) 
            # fq_raw = self.decoder(fq_raw_, word_raw, pad_mask_raw, process_raw_feature, process_raw_feature_mask ,style_embed)
            
            # fq_raw, fq_txt_raw = self.decoder(fq_raw_, word_raw, pad_mask_raw, process_raw_feature, process_raw_feature_mask)
            fq_raw = self.decoder(fq_raw_, word_raw, pad_mask_raw, process_raw_feature, process_raw_feature_mask)
            fq_raw = fq_raw.reshape(b, c, h, w)
          
            # b, 1, 104, 104
            pred_raw = self.proj(fq_raw, state_raw)
            return pred_raw.detach()
        else:
            if word.shape[1] > self.word_length:
                # 双分支
                text_raw = word[:, :self.word_length]
                text_en  = word[:, self.word_length:]
                pad_mask_raw = torch.zeros_like(text_raw).masked_fill_(text_raw == 0, 1).bool()
                pad_mask_en  = torch.zeros_like(text_en).masked_fill_(text_en == 0, 1).bool()

                vis = self.backbone.encode_image(img)
                word_raw, word_en, state_raw, state_en = self.backbone.encode_text(word, self.word_length)

                process_en_feature = None
                process_en_feature_mask = None
                process_raw_feature = None
                process_raw_feature_mask = None

                if en_labels is not None:
                    # raw 解耦
                    # process_raw_feature, process_raw_feature_mask = process_en(word_raw, en_labels_raw)
                    process_raw_feature, process_raw_feature_mask = process_en(
                    word_raw,
                    en_labels_raw,
                    parser_noise_ratio=0.0 if self.training else self.parser_noise_ratio,
                    parser_noise_mode=self.parser_noise_mode,
                    token_valid_mask=~pad_mask_raw,
                )
                    process_raw_feature_mask = process_raw_feature_mask.bool()
                    process_raw_feature = self._apply_soft_parser_prior(
                        word_raw, process_raw_feature, process_raw_feature_mask)
                    state_raw_ = state_raw.unsqueeze(1)
                    process_raw_feature = torch.cat([process_raw_feature, state_raw_.expand(-1, process_raw_feature.shape[1], -1)], dim=-1)
                    linear = self.linear.to(process_raw_feature.device)
                    process_raw_feature = process_raw_feature.to(dtype=next(linear.parameters()).dtype)
                    process_raw_feature = linear(process_raw_feature)

                    # en 解耦
                    # process_en_feature, process_en_feature_mask = process_en(word_en, en_labels)
                    process_en_feature, process_en_feature_mask = process_en(
                        word_en,
                        en_labels,
                        parser_noise_ratio=0.0 if self.training else self.parser_noise_ratio,
                        parser_noise_mode=self.parser_noise_mode,
                        token_valid_mask=~pad_mask_en,
                    )
                    process_en_feature_mask = process_en_feature_mask.bool()
                    process_en_feature = self._apply_soft_parser_prior(
                        word_en, process_en_feature, process_en_feature_mask)
                    state_en_ = state_en.unsqueeze(1)
                    process_en_feature = torch.cat([process_en_feature, state_en_.expand(-1, process_en_feature.shape[1], -1)], dim=-1)
                    process_en_feature = process_en_feature.to(dtype=next(linear.parameters()).dtype)
                    process_en_feature = linear(process_en_feature)

                if process_en_feature is not None and process_en_feature.dtype != word_en.dtype:
                    process_en_feature = process_en_feature.to(dtype=word_en.dtype)
                    process_raw_feature = process_raw_feature.to(dtype=word_raw.dtype)

                fq_raw_ = self.neck(vis, state_raw)
                fq_en_  = self.neck(vis, state_en)

                b, c, h, w = fq_raw_.size()
                fq_raw = self.decoder(fq_raw_, word_raw, pad_mask_raw, process_raw_feature, process_raw_feature_mask)
                fq_en  = self.decoder(fq_en_, word_en, pad_mask_en, process_en_feature, process_en_feature_mask)
                fq_raw__ = fq_raw.reshape(b, c, h, w)
                fq_en__  = fq_en.reshape(b, c, h, w)

                pred_raw = self.proj(fq_raw__, state_raw)
                pred_en  = self.proj(fq_en__, state_en)

                if self.training:
                    if pred_raw.shape[-2:] != mask.shape[-2:]:
                        mask = F.interpolate(mask, pred_raw.shape[-2:], mode='nearest').detach()
                    loss_raw = F.binary_cross_entropy_with_logits(pred_raw, mask)
                    loss_en  = F.binary_cross_entropy_with_logits(pred_en, mask)
                    loss = {'loss_raw': loss_raw, 'loss_en': loss_en}
                    return pred_raw.detach(), mask, loss
                else:
                    return pred_raw.detach(), pred_en.detach()

            else:
                # 单分支
                text_raw = word
                pad_mask_raw = torch.zeros_like(text_raw).masked_fill_(text_raw == 0, 1).bool()

                vis = self.backbone.encode_image(img)
                word_raw, state_raw = self.backbone.encode_text(word, self.word_length,if_infer=True)

                process_raw_feature = None
                process_raw_feature_mask = None
                if en_labels_raw is not None:
                    # process_raw_feature, process_raw_feature_mask = process_en(word_raw, en_labels_raw)
                    process_raw_feature, process_raw_feature_mask = process_en(
                        word_raw,
                        en_labels_raw,
                        parser_noise_ratio=0.0 if self.training else self.parser_noise_ratio,
                        parser_noise_mode=self.parser_noise_mode,
                        token_valid_mask=~pad_mask_raw,
                    ) 
                    process_raw_feature_mask = process_raw_feature_mask.bool()
                    process_raw_feature = self._apply_soft_parser_prior(
                        word_raw, process_raw_feature, process_raw_feature_mask)
                    state_raw_ = state_raw.unsqueeze(1)
                    process_raw_feature = torch.cat([process_raw_feature, state_raw_.expand(-1, process_raw_feature.shape[1], -1)], dim=-1)
                    linear = self.linear.to(process_raw_feature.device)
                    process_raw_feature = process_raw_feature.to(dtype=next(linear.parameters()).dtype)
                    process_raw_feature = linear(process_raw_feature)

                fq_raw_ = self.neck(vis, state_raw)
                b, c, h, w = fq_raw_.size()
                fq_raw = self.decoder(fq_raw_, word_raw, pad_mask_raw, process_raw_feature, process_raw_feature_mask)
                fq_raw__ = fq_raw.reshape(b, c, h, w)
                pred_raw = self.proj(fq_raw__, state_raw)

                if self.training:
                    if pred_raw.shape[-2:] != mask.shape[-2:]:
                        mask = F.interpolate(mask, pred_raw.shape[-2:], mode='nearest').detach()
                    loss_raw = F.binary_cross_entropy_with_logits(pred_raw, mask)
                    loss = {'loss_raw': loss_raw}
                    return pred_raw.detach(), mask, loss
                else:
                    return pred_raw.detach()
