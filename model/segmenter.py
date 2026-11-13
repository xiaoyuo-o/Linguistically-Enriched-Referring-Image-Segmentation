import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model

from .layers import FPN, Projector, TransformerDecoder
from tools.losses import *
from tools.process_caption_parse import *
from info_nce import InfoNCE
import torch.nn.functional as F
import math


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


def simsiam_loss(p, z):
    # z 是目标，不反传
    z = z.detach()
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return - (p * z).sum(dim=-1).mean()



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
    
class TextProjector(nn.Module):
    def __init__(self, input_dim, vis_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),  # 可选：加深网络
            nn.ReLU(),
            nn.Linear(input_dim, vis_dim)   # 最终输出为 token 序列长度 L
        )

    def forward(self, x):
        """
        input:
        x: [B,L,1024]
        output:
        x: [B,L,vis_dim]
        """
        B, L, d = x.shape
        x = x.reshape(B*L,d)
        out = self.mlp(x)
        out = out.view(B,L,-1)
        return out

class VisionGuidedTextAlign(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.cross_attn_1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.txt_proj = nn.Linear(embed_dim, embed_dim)
        self.vis_proj = nn.Linear(embed_dim*2, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.txt_proj_fuse = nn.Linear(embed_dim*2, embed_dim)
        self.vis_gate_proj = nn.Sequential(
    nn.Linear(embed_dim, embed_dim),
    nn.ReLU(),
    nn.Linear(embed_dim, 1),
    nn.Sigmoid()
)
    @staticmethod
    def pos1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe.unsqueeze(1)  # n, 1, 512

    @staticmethod
    def pos2d(d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512
    
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)
    
    def forward(self, txt1, txt2, txt1_pos, txt2_pos,
                vis, txt1_pad_mask=None, txt2_pad_mask=None):
        B, C, H, W = vis.size()

        # position encoding
        vis_pos = self.pos2d(C, H, W)
        # print(vis_pos.shape)
        # 加入位置编码
        txt1_in = txt1 + txt1_pos
        txt2_in = txt2 + txt2_pos
        vis = vis.reshape(B, C, -1).permute(2, 0, 1)   #676,b,c
        # vis_pos = vis_pos.permute(1,0,2)
        vis_with_pos = self.with_pos_embed(vis, vis_pos)  #676,b,c
        vis_in = self.vis_proj(vis_with_pos.permute(1, 0, 2))  # [B, 676, 512]
        # Vision-guided attention key/value: 用视觉特征指引 cross-attention
        query1 = self.txt_proj(txt2_in)
        key_1 = self.txt_proj(txt1_in)


        #双向： 
        query2 = self.txt_proj(txt1_in)
        key_2 = self.txt_proj(txt2_in)
        txt1_updated, _ = self.cross_attn_2(query=query2,
                                            key=key_2,
                                            value=txt2,
                                            key_padding_mask=txt2_pad_mask)
        gate2 = torch.sigmoid(self.vis_gate_proj(vis_in.mean(1, keepdim=True)))
        
        # query = self.txt_proj(txt1_in) + vis_in.mean(1, keepdim=True)
        # key_1 = self.txt_proj(txt2_in) + vis_in.mean(1, keepdim=True)  # 可选：将图像引入 key
        # key_2 = self.txt_proj(txt1_in) + vis_in.mean(1, keepdim=True)
        # key_1 = self.txt_proj(txt2_in)
        # Cross Attention: txt1 attends to txt2 (guided by vision)
        txt2_updated, _ = self.cross_attn_1(query=query1,
                                            key=key_1,
                                            value=txt1,
                                            key_padding_mask=txt1_pad_mask)
        gate1 = torch.sigmoid(self.vis_gate_proj(vis_in.mean(1, keepdim=True)))  # [B, 1, C]
        txt2_updated = txt2 + gate1 * txt2_updated       
        txt2_updated = self.norm1(txt2_updated)

        # Cross Attention: txt2 attends to txt1 (guided by vision)
        # txt2_updated, _ = self.cross_attn_2(query=txt2_in,
        #                                     key=key_2,
        #                                     value=txt1,
        #                                     key_padding_mask=txt1_pad_mask)
        # txt2_updated = self.norm2(txt2 + txt2_updated)  #b,n,dim
        
        # txt_updated = torch.concat([txt1_updated, txt2_updated], dim=-1)
        # txt_updated = self.txt_proj_fuse(txt_updated)
        return txt2_updated
    

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


        # self.state_fuse_linear = nn.Linear(cfg.word_dim*2,cfg.word_dim)
        # self.vis_guided_proj = nn.Conv2d(in_channels=cfg.word_dim*2, out_channels=cfg.word_dim, kernel_size=1)
        # #两种text做交互的模块：
        # self.vision_guided_text_align = VisionGuidedTextAlign(embed_dim=cfg.word_dim, num_heads=cfg.num_head)
        #写一个能够判断是否是简单还是复杂文本的MLP

        self.info_nce = InfoNCE(negative_mode='paired')
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
