import os
from typing import List, Union, Tuple, Dict
import ftfy
import regex as re
import html
import cv2
import lmdb
import numpy as np
import pickle
import torch
import sng_parser
from torch.utils.data import Dataset

from .simple_tokenizer import SimpleTokenizer as _Tokenizer

info = {
    'refcoco': {
        'train': 42404,
        'val': 3811,
        'val-test': 3811,
        'testA': 1975,
        'testB': 1810
    },
    'refcoco+': {
        'train': 42278,
        'val': 3805,
        'val-test': 3805,
        'testA': 1975,
        'testB': 1798
    },
    'refcocog_u': {
        'train': 42226,
        'val': 2573,
        'val-test': 2573,
        'test': 5023
    },
    'refcocog_g': {
        'train': 44822,
        'val': 5000,
        'val-test': 5000
    }
}
_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
def tokenize_en(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> Tuple[torch.LongTensor, List[Dict[int, int]]]:

    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = []
    char_to_token_map_list = []

    for text in texts:
        # text = text.replace('—', '-')  # 替换 em-dash

        cleaned_text = whitespace_clean(basic_clean(text)).lower()
        

        # 按正则拆成 token
        tokens = re.findall(_tokenizer.pat, cleaned_text)

        bpe_tokens = []
        char_to_token = {}
        current_char = 0  # 当前 token 起始的原始字符位置
        token_idx = 1  # 从1开始，因为0是 <|startoftext|>

        for token in tokens:
            if token != ' ':
                token = ''.join(_tokenizer.byte_encoder[b] for b in token.encode('utf-8'))
                bpe_result = _tokenizer.bpe(token).split(' ')
                # for bpe_token in bpe_result:
                #     if bpe_token not in _tokenizer.encoder:
                #         print(f"[Warning] Token not found in encoder: {bpe_token}")
                #         continue
                
                token_ids = [_tokenizer.encoder[bpe_token] for bpe_token in bpe_result]

                bpe_tokens.extend(token_ids)

            # 映射原始字符到 token index（注意：映射的是字符 index → 第几个 token）
            for i in range(len(token)):
                char_to_token[current_char + i] = token_idx
            if token == ' ':
                token_idx = token_idx
            else:
                token_idx += len(token_ids)

            current_char += len(token)

        final_token_ids = [sot_token] + bpe_tokens + [eot_token]
        all_tokens.append(final_token_ids)
        char_to_token_map_list.append(char_to_token)

    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result, char_to_token_map_list

def get_positive_map(tokens_positive, max_token, char_to_token_list):
    positive_map = torch.zeros((len(tokens_positive), max_token), dtype=torch.float)  # ([positive], 256])，len(tokens_positive)实际上是定好的最大物体数量，在这我们只让他为1


    for j, tok_list in enumerate(tokens_positive):
        (beg, end) = tok_list   #两个坐标，起始值与结束的字符的位置
        beg = int(beg)
        end = int(end)
        beg_pos = char_to_token_list[0][beg]
        end_pos = char_to_token_list[0][end-1]
        if beg_pos is None:
            try:
                beg_pos = char_to_token_list[0][beg+1]
                if beg_pos is None:
                    beg_pos = char_to_token_list[0][beg+2]
            except:
                beg_pos = None
        if end_pos is None:
            try:
                end_pos = char_to_token_list[0][end-2]
                if end_pos is None:
                    end_pos = char_to_token_list[0][end-3]
            except:
                end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        positive_map[j, beg_pos:end_pos + 1].fill_(1)

    positive_map = positive_map / (positive_map.sum(-1)[:, None] + 1e-12)
    return positive_map.numpy()   #[num_query,60]

def generate_anno(caption):
    anno ={}
    graph_node, graph_edge = sng_parser.parse(caption)
    anno["graph_node"] = graph_node
    anno["graph_edge"] = graph_edge
    auxi_entity = None
    for node in graph_node:
        if (node["node_id"] != 0) and (node["node_type"] == "Object"):
            auxi_entity = node
            break
    anno["auxi_entity"] = auxi_entity            
        #caption_en decoupling
    anno["utterance"] = caption
    return anno

def get_token_positive_map_by_parse(anno, max_token, char_to_token_list, MAX_NUM_OBJ=5):

    # node and edge
    graph_node = anno["graph_node"]

    # step main/modify(attri)/pron/other(auxi)/rel
    target_char_span = np.zeros((MAX_NUM_OBJ, 2))   # target(main)
    modify_char_span = np.zeros((MAX_NUM_OBJ, 2))   # modify(attri)
    pron_char_span = np.zeros((MAX_NUM_OBJ, 2))     # pron
    rel_char_span = np.zeros((MAX_NUM_OBJ, 2))      # rel
    if graph_node[0]['node_id'] != 0:
        print(1) 
    assert graph_node[0]['node_id'] == 0   #这里替换成找graph_node[i]["node_id"] == 0，找i
    main_entity_target = graph_node[0]['target_char_span']
    main_entity_modify = graph_node[0]['mod_char_span']
    main_entity_pron   = graph_node[0]['pron_char_span']
    main_entity_rel   = graph_node[0]['rel_char_span']

    # other(auxi) object token
    other_target_char_span = np.zeros((MAX_NUM_OBJ, 2))
    other_entity_target = []
    if len(graph_node) > 1:
        for node in graph_node:
            if node["node_id"] != 0 and node["node_type"] == "Object":
                for span in node['target_char_span']:
                    other_entity_target.append(span)

    num_t = 0
    num_m = 0
    num_p = 0
    num_o = 0
    num_r = 0
    # target(main obj.) token
    for t, target in enumerate(main_entity_target):
        if t+1 > MAX_NUM_OBJ:
            break
        target_char_span[t] = target
        num_t = t+1
    # modify(attribute) token
    for m, modify in enumerate(main_entity_modify):
        if m+1 >MAX_NUM_OBJ:
            break
        modify_char_span[m] = modify
        num_m = m+1
    # pron token
    for p, pron in enumerate(main_entity_pron):
        if p+1 > MAX_NUM_OBJ:
            break
        pron_char_span[p] = pron
        num_p = p+1
    for o, other in enumerate(other_entity_target):
        if o+1 > MAX_NUM_OBJ:
            break
        other_target_char_span[o] = other
        num_o = o+1
    # rel token add 0727
    for r, rel in enumerate(main_entity_rel):
        if r+1 > MAX_NUM_OBJ:
            break
        rel_char_span[r] = rel
        num_r = r+1

    target_positive_map = np.zeros((MAX_NUM_OBJ, max_token))
    modify_positive_map = np.zeros((MAX_NUM_OBJ, max_token))
    pron_positive_map = np.zeros((MAX_NUM_OBJ, max_token))
    other_entity_positive_map = np.zeros((MAX_NUM_OBJ, max_token))
    rel_positive_map = np.zeros((MAX_NUM_OBJ, max_token))
    gt_map_t = get_positive_map(target_char_span[:num_t], max_token, char_to_token_list)
    gt_map_m = get_positive_map(modify_char_span[:num_m], max_token, char_to_token_list)
    gt_map_p = get_positive_map(pron_char_span[:num_p], max_token, char_to_token_list)
    gt_map_o = get_positive_map(other_target_char_span[:num_o], max_token, char_to_token_list)
    gt_map_r = get_positive_map(rel_char_span[:num_r], max_token, char_to_token_list)
    
    gt_map_t = gt_map_t.sum(axis=0)
    gt_map_m = gt_map_m.sum(axis=0)
    gt_map_p = gt_map_p.sum(axis=0)
    gt_map_o = gt_map_o.sum(axis=0)
    gt_map_r = gt_map_r.sum(axis=0)

    # NOTE text position label
    target_positive_map[:1] = gt_map_t          # main object
    modify_positive_map[:1] = gt_map_m          # attribute
    pron_positive_map[:1]   = gt_map_p          # pron
    other_entity_positive_map[:1] = gt_map_o    # auxi obj
    rel_positive_map[:1]   = gt_map_r           # relation



    return target_char_span, target_positive_map, modify_positive_map, pron_positive_map, \
        other_entity_positive_map, rel_positive_map

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text





def loads_pickle(buf):
    """
    Args:
        buf: the output of `dumps` from pickle.
    """
    return pickle.loads(buf)

def dumps_pickle(obj):
    """
    Args:
        obj: The object to be serialized.
    """
    return pickle.dumps(obj)



class RefDataset(Dataset):
    def __init__(self, lmdb_dir, mask_dir, dataset, split, mode, input_size,
                 word_length, word_en_length, new_lmdb_dir, is_decouple=False):
        super(RefDataset, self).__init__()
        self.lmdb_dir = lmdb_dir
        self.new_lmdb_dir = new_lmdb_dir
        self.mask_dir = mask_dir
        self.dataset = dataset
        self.split = split
        self.mode = mode
        self.input_size = (input_size, input_size)
        self.word_length = word_length
        self.word_en_length = word_en_length
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        self.length = info[dataset][split]
        self.env = None
        self.env_new = None
        self.is_decouple = is_decouple
       

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_dir,
                             subdir=os.path.isdir(self.lmdb_dir),
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pickle(txn.get(b'__len__'))
            self.keys = loads_pickle(txn.get(b'__keys__'))
    def _init_db_new(self):
        self.env_new = lmdb.open(self.new_lmdb_dir,
                                 subdir=os.path.isdir(self.lmdb_dir),
                                 readonly=True,
                                 lock=False,
                                 readahead=False,
                                 meminit=False)
        with self.env_new.begin(write=False) as txn_new:
            self.length_new = loads_pickle(txn_new.get(b'__len__'))
            self.keys_new = loads_pickle(txn_new.get(b'__keys__'))

    def __len__(self):
        return self.length
    


    def __getitem__(self, index):
        # Delay loading LMDB data until after initialization: https://github.com/chainer/chainermn/issues/129
        if self.env is None:
            self._init_db()
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        ref = loads_pickle(byteflow)
        # img
        ori_img = cv2.imdecode(np.frombuffer(ref['img'], np.uint8),
                               cv2.IMREAD_COLOR)
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img_size = img.shape[:2]
        # mask
        seg_id = ref['seg_id']
        mask_dir = os.path.join(self.mask_dir, str(seg_id) + '.png')
        # sentences
        idx = np.random.choice(ref['num_sents'])
        sents = ref['sents']
        # transform
        mat, mat_inv = self.getTransformMat(img_size, True)
        img = cv2.warpAffine(
            img,
            mat,
            self.input_size,
            flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])
        # if self.mode == 'train':
        if self.env_new is None:
            self._init_db_new()
        env_new = self.env_new
        with env_new.begin(write=False) as txn_new:
            byteflow_new = txn_new.get(self.keys_new[index])
        ref_new = loads_pickle(byteflow_new)

        assert ref['seg_id'] == ref_new['seg_id'], f"Mismatch between segment IDs! Dataset 1: {ref['seg_id']}, Dataset 2: {ref_new['seg_id']}"
        # mask transform
        sents_new = ref_new['sents']
        idx_new = np.random.choice(ref_new['num_sents'])
        if self.mode == 'train':
            mask = cv2.imdecode(np.frombuffer(ref['mask'], np.uint8),
                                cv2.IMREAD_GRAYSCALE)
            mask = cv2.warpAffine(mask,
                                mat,
                                self.input_size,
                                flags=cv2.INTER_LINEAR,
                                borderValue=0.)
            mask = mask / 255.
            # sentence -> vector
            sent = sents[idx]
            caption_raw = ' '.join(sent.replace(',', ' ,').split())           
            sent_new = sents_new[idx_new]
            caption_new = ' '.join(sent_new.replace(',', ' ,').split())
            word_vec, char_to_token_map_list_raw = tokenize_en(caption_raw, self.word_length, True)
            word_vec_new, char_to_token_map_list = tokenize_en(caption_new, self.word_en_length, True)
            word_vec = word_vec.squeeze(0)
            word_vec_new = word_vec_new.squeeze(0)
            combined_word_vec = torch.cat([word_vec, word_vec_new], dim=0)
            if self.is_decouple:
                #对原来的解耦
                anno_raw = generate_anno(caption_raw)
                tokens_positive_raw, positive_map_raw, modify_positive_map_raw, pron_positive_map_raw, \
                other_entity_map_raw, rel_positive_map_raw = get_token_positive_map_by_parse(anno_raw, self.word_length, char_to_token_map_list_raw)
                en_labels_raw = {
                'tokens_positive': tokens_positive_raw.astype(np.int64),
                'positive_map': positive_map_raw,
                'modify_positive_map': modify_positive_map_raw,
                'pron_positive_map': pron_positive_map_raw,
                'other_entity_map': other_entity_map_raw,
                'rel_positive_map': rel_positive_map_raw
                }

                #对enhance text解耦
                
                anno = generate_anno(caption_new)
                tokens_positive, positive_map, modify_positive_map, pron_positive_map, \
                other_entity_map, rel_positive_map = get_token_positive_map_by_parse(anno, self.word_en_length, char_to_token_map_list)
                en_labels = {
                'tokens_positive': tokens_positive.astype(np.int64),
                'positive_map': positive_map,
                'modify_positive_map': modify_positive_map,
                'pron_positive_map': pron_positive_map,
                'other_entity_map': other_entity_map,
                'rel_positive_map': rel_positive_map
                }
                img, mask = self.convert(img, mask)
                return img, combined_word_vec, mask, en_labels, en_labels_raw
            else:
                img, mask = self.convert(img, mask)
                return img, combined_word_vec, mask
        elif self.mode == 'val':
            # sentence -> vector
            sent = sents[0]
            sent_new = sents_new[0]
            caption_raw = ' '.join(sent.replace(',', ' ,').split())
            caption_new = ' '.join(sent_new.replace(',', ' ,').split())
            word_vec, char_to_token_map_list_raw = tokenize_en(sent, self.word_length, True)
            word_vec = word_vec.squeeze(0)
            word_vec_new, char_to_token_map_list = tokenize_en(caption_new, self.word_en_length, True)
            word_vec_new = word_vec_new.squeeze(0)
            combined_word_vec = torch.cat([word_vec,word_vec_new], dim=0)

            if self.is_decouple:
                anno_raw = generate_anno(caption_raw)
                tokens_positive_raw, positive_map_raw, modify_positive_map_raw, pron_positive_map_raw, \
                other_entity_map_raw, rel_positive_map_raw = get_token_positive_map_by_parse(anno_raw, self.word_length, char_to_token_map_list_raw)
                en_labels_raw = {
                'tokens_positive': tokens_positive_raw.astype(np.int64),
                'positive_map': positive_map_raw,
                'modify_positive_map': modify_positive_map_raw,
                'pron_positive_map': pron_positive_map_raw,
                'other_entity_map': other_entity_map_raw,
                'rel_positive_map': rel_positive_map_raw
                }
                anno = generate_anno(caption_new)
                tokens_positive, positive_map, modify_positive_map, pron_positive_map, \
                other_entity_map, rel_positive_map = get_token_positive_map_by_parse(anno, self.word_en_length, char_to_token_map_list)
                en_labels = {
                'tokens_positive': tokens_positive.astype(np.int64),
                'positive_map': positive_map,
                'modify_positive_map': modify_positive_map,
                'pron_positive_map': pron_positive_map,
                'other_entity_map': other_entity_map,
                'rel_positive_map': rel_positive_map
                }

                img = self.convert(img)[0]
                params = {
                    'mask_dir': mask_dir,
                    'inverse': mat_inv,
                    'ori_size': np.array(img_size)
                }
                return img, combined_word_vec, params, en_labels, en_labels_raw
            else:
                img = self.convert(img)[0]
                params = {
                    'mask_dir': mask_dir,
                    'inverse': mat_inv,
                    'ori_size': np.array(img_size)
                }
                return img, combined_word_vec, params

        else:
            # sentence -> vector
            img = self.convert(img)[0]
            params = {
                'ori_img': ori_img,
                'seg_id': seg_id,
                'mask_dir': mask_dir,
                'inverse': mat_inv,
                'ori_size': np.array(img_size),
                'sents': sents
            }
            is_decouple = self.is_decouple
            return img, params, is_decouple

    def getTransformMat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None

    def convert(self, img, mask=None):
        # Image ToTensor & Normalize
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)
        # Mask ToTensor
        if mask is not None:
            mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()
        return img, mask

    def __repr__(self):
        return self.__class__.__name__ + "(" + \
            f"db_path={self.lmdb_dir}, " + \
            f"dataset={self.dataset}, " + \
            f"split={self.split}, " + \
            f"mode={self.mode}, " + \
            f"input_size={self.input_size}, " + \
            f"word_length={self.word_length}"

    # def get_length(self):
    #     return self.length

    # def get_sample(self, idx):
    #     return self.__getitem__(idx)
