import os
from typing import List, Union, Tuple, Dict
import ftfy
import regex as re
import html
import cv2
import lmdb
import numpy as np
from PIL import Image, ImageFilter
import pickle
import torch
import sng_parser
from tools import refer
from torch.utils.data import Dataset
import albumentations as A
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
import math
import random
from torchvision.utils import save_image
def normalize(img_w, img_h, box, tail=True):
    xmin, ymin, xmax, ymax = box[:4]   #左下角和右上角
    new_box = [xmin/img_w, ymin/img_h, xmax/img_w, ymax/img_h]
    new_box = np.clip(new_box, 0, 1).tolist()
    if tail and len(box)>=4:
        new_box.extend(box[4:])
    return new_box



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
                 word_length, word_en_length, new_lmdb_dir, args, is_decouple=False):
        super(RefDataset, self).__init__()
        self.lmdb_dir = lmdb_dir
        self.new_lmdb_dir = new_lmdb_dir
        self.mask_dir = mask_dir
        self.dataset = dataset
        self.split = split
        self.mode = mode

        self.img_sz = input_size  #

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
        self.args = args  #

        self.is_decouple = is_decouple

       #消融实验的一些设置：
        self.if_two_branches = args.if_two_branches


        self.resize_bg1 = A.Compose([
            A.Resize(input_size, input_size, always_apply=True)
        ])

        if self.mode == 'train':
            #  首先先实例化类
            ## Bringing out logits
            if self.dataset == 'refcoco' :
                self.refer = refer.REFER(data_root=args.data_root, dataset='refcoco', splitBy='unc')
            elif self.dataset == 'refcoco+' :
                self.refer = refer.REFER(data_root=args.data_root, dataset='refcoco+', splitBy='unc')
            elif self.dataset  =='refcocog_u' :
                self.refer = refer.REFER(data_root=args.data_root, dataset='refcocog', splitBy='umd')
            print(f"Bringing out logits of {self.dataset} dataset")


            ## Tools by refer.REFER
            #  得到instance的标号
            ref_ids = self.refer.getRefIds(split=split)
            #  通过ref_ids得到对应的这个img_id
            self.ref_id2idx = dict(zip(ref_ids, range(len(ref_ids)))) # ref_id -> idx(key)
            self.idx2ref_id = dict(zip(range(len(ref_ids)), ref_ids)) # idx(key) -> ref_id

        else :
            print("Test mode does not require logits of dataset")
        if self.args.blur:
            self.blur = ImageFilter.GaussianBlur(100)
        np.random.seed()



    def _init_db(self):
        self.env = lmdb.open(
            self.lmdb_dir,
            subdir=os.path.isdir(self.lmdb_dir),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.length = loads_pickle(txn.get(b'__len__'))
            self.keys = loads_pickle(txn.get(b'__keys__'))

        if self.mode != 'train':
            return

        self.t2i_env = lmdb.open(
            self.args.t2i_similarity_lmdb,
            subdir=True,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.t2i_env.begin(write=False) as txn:
            self.logit_keys = loads_pickle(txn.get(b'__keys__'))

    def _init_db_new(self):
        self.env_new = lmdb.open(
            self.new_lmdb_dir,
            subdir=os.path.isdir(self.lmdb_dir),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env_new.begin(write=False) as txn_new:
            self.keys_new = loads_pickle(txn_new.get(b'__keys__'))

        if self.mode != 'train':
            return

        self.t2i_env_en = lmdb.open(
            self.args.enhanced_t2i_similarity_lmdb,
            subdir=True,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.t2i_env_en.begin(write=False) as txn:
            self.logit_keys_en = loads_pickle(txn.get(b'__keys__'))

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __getitem__(self, index):
        # Delay loading LMDB data until after initialization: https://github.com/chainer/chainermn/issues/129
        if self.env is None:
            self._init_db()
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        ref = loads_pickle(byteflow)


        # 只有两分支时才加载增强分支
        if self.if_two_branches:
            if self.env_new is None:
                self._init_db_new()
            env_new = self.env_new
            with env_new.begin(write=False) as txn_new:
                byteflow_new = txn_new.get(self.keys_new[index])
            ref_new = loads_pickle(byteflow_new)
            assert ref['seg_id'] == ref_new['seg_id'], f"Mismatch between segment IDs! Dataset 1: {ref['seg_id']}, Dataset 2: {ref_new['seg_id']}"


        if self.mode == 'train':
            ref_id = self.idx2ref_id[index]
            t2i_index = self.ref_id2idx[ref_id]


            with self.t2i_env.begin(write=False) as txn:
                bytes = str(self.logit_keys[t2i_index]).encode("utf-8")
                t2i_byteflow = txn.get(bytes)
            t2i_similarity = loads_pickle(t2i_byteflow)


            if self.if_two_branches:
                with self.t2i_env_en.begin(write=False) as txn_en:
                    bytes = str(self.logit_keys_en[t2i_index]).encode("utf-8")
                    t2i_byteflow_en = txn_en.get(bytes)
                t2i_similarity_en = loads_pickle(t2i_byteflow_en)


            target_sent_idx = np.random.choice(list(range(ref['num_sents'])), size=1, replace=False)[0]
            sent = ref['sents'][target_sent_idx]
            if self.if_two_branches:
                target_sent_idx_en = np.random.choice(list(range(ref_new['num_sents'])), size=1, replace=False)[0]

            # sent_new = sents_new[target_sent_idx_en]
            #mosic的图像的个数，即K
            if self.args.num_bgs > 1:
                aug_prob = self.args.aug_prob

                # Before retrieval_epoch: One Image or Random Mosaic
                if self.current_epoch < self.args.retrieval_epoch:
                    if self.args.mix_grid :
                        num_bgs = np.random.choice([1, 4, 9], p=[1-aug_prob, aug_prob/2, aug_prob/2]) #
                        mosaic_type = 'random'
                    else :
                        num_bgs = np.random.choice([1, 4], p=[1-aug_prob, aug_prob]) #
                        mosaic_type = 'random'
                # Retrieval
                else:
                    rand_prob = self.args.rand_prob
                    retr_prob = self.args.retr_prob
                    # After retrieval_epoch: Decide between One Image, Random Mosaic, or Retrieval Based Mosaic
                    choice = np.random.choice(['one', 'random', 'retrieval'], p=[1-(rand_prob + retr_prob), rand_prob, retr_prob])
                    mosaic_type = choice
                    if choice == 'one':
                        num_bgs = 1
                    elif self.args.mix_grid :
                        num_bgs = np.random.choice([4, 9], p=[0.5, 0.5])
                    else :
                        num_bgs = 4
            else:
                num_bgs = 1
                mosaic_type = 'one'
        ## Test, Val Config
        else :
            num_bgs = 1
            mosaic_type ='one'
        ## Choosing dataloader strategy
        if num_bgs > 1 :
            if mosaic_type == 'retrieval':
                sent_id = list(t2i_similarity.keys())[target_sent_idx]
                candidate_pool = t2i_similarity[sent_id]
                if self.if_two_branches:
                    sent_id_en = list(t2i_similarity_en.keys())[target_sent_idx_en]
                    candidate_pool = random.choice(
                        [candidate_pool, t2i_similarity_en[sent_id_en]]
                    )

                invalid_choices = {
                    img_id
                    for img_id, score in candidate_pool
                    if score > self.args.t2i_sim_thres
                }
                valid_img_score_list = [
                    img_id
                    for img_id, _ in candidate_pool
                    if img_id not in invalid_choices
                ]
                len_valid = len(valid_img_score_list)

                if len_valid == 0:
                    print("Valid image list is 0", len(invalid_choices))
                    valid_img_score_list = [img_id for img_id, _ in candidate_pool[-300:]]
                    len_valid = len(valid_img_score_list)

                if self.args.top_k < 20 :
                    img_ids = list(np.random.choice(valid_img_score_list[:self.args.top_k], size=num_bgs-1, replace=False))
                elif len_valid < self.args.top_k :
                    img_ids = list(np.random.choice(valid_img_score_list, size=num_bgs-1, replace=True))
                else :
                    img_ids = list(np.random.choice(valid_img_score_list[:self.args.top_k], size=num_bgs-1, replace=True))

                ref_int = [self.refer.imgToRefs[i][0]['ref_id'] for i in img_ids]
                keys = [str(self.ref_id2idx[k]).encode('utf-8') for k in ref_int]

            else:  # mosaic_type == 'random'
                # Random Mosaic
                keys = list(np.random.choice(self.keys, size=num_bgs-1, replace=False))

            refs = []
            for key in keys:
                with env.begin(write=False) as txn:
                    byteflow = txn.get(key)
                    ref_other = loads_pickle(byteflow)
                refs.append(ref_other)
        ## One Image (num_bgs = 1)
        else:
            keys = []
            refs = []

        if num_bgs == 1:
            insert_idx = 0
        else:
            insert_idx = np.random.choice(range(num_bgs))


        refs.insert(insert_idx, ref)   #如果是测试的时候，就只有一个


        # if self.args.aug.tgt_selection == 'fixed':
        target_idx = insert_idx #验证或者测试，target_idx=0



        # load items
        imgs, masks, sents_arr, seg_ids = [], [], [], []
        org_img_sizes = []
        for ref in refs:

            ori_img = cv2.imdecode(np.frombuffer(ref['img'], np.uint8),
                                cv2.IMREAD_COLOR)
            img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
            org_img_sizes.append(img.shape[:2])

            mask = cv2.imdecode(np.frombuffer(ref['mask'], np.uint8),
                    cv2.IMREAD_GRAYSCALE)
            mask = mask / 255.
            masks.append(mask)

            seg_id = ref['seg_id']
            seg_ids.append(seg_id)
            sents_arr.append(ref['sents'])


        # image resize and apply 4in1 augmentation
        if num_bgs==1:
            resized = self.resize_bg1(image=imgs[0], mask=masks[0])
            imgs, masks = [resized['image']], [resized['mask']]
            img = imgs[0]
        else:
            if self.args.move_crs_pnt:
                if self.args.rand_crs :
                    print("Cross point random shift")
                    crs_y = np.random.randint(0, self.img_sz+1)
                    crs_x = np.random.randint(0, self.img_sz+1)
                else :
                    print("Cross point shift within range")
                    quarter_sz = self.each_img_sz/2
                    crs_y = np.random.randint(quarter_sz, self.img_sz+1-quarter_sz)
                    crs_x = np.random.randint(quarter_sz, self.img_sz+1-quarter_sz)
            else:
                denum = int(math.sqrt(num_bgs))
                crs_y = self.img_sz//denum
                crs_x = self.img_sz//denum

            imgs_resized = []
            masks_resized = []

            for i in range(num_bgs):
                row = i // denum
                col = i % denum
                y_start = row * crs_y
                y_end = y_start + crs_y
                x_start = col * crs_x
                x_end = x_start + crs_x

                if y_end > self.img_sz or x_end > self.img_sz:
                    img_resized = np.zeros([crs_y, crs_x, 3])
                    mask_resized = np.zeros([crs_y, crs_x])
                else:
                    resize_transform = A.Compose([
                        A.Resize(crs_y, crs_x, always_apply=True)
                    ])
                    temp = resize_transform(image=imgs[i], mask=masks[i])
                    img_resized = temp['image']
                    mask_resized = temp['mask']

                imgs_resized.append(img_resized)
                masks_resized.append(mask_resized)

            imgs = imgs_resized
            masks = masks_resized


            # scale effect ablation
            if self.args.blur:
                imgs = [np.asarray(Image.fromarray(x).filter(self.blur)) if i!=insert_idx else x for i, x in enumerate(imgs)]

            num_rows = num_cols = int(math.sqrt(num_bgs))
            idxs = [(i*num_cols,i*num_cols+num_cols) for i in range(num_rows)]
            img = [np.concatenate(imgs[_from:_to], axis=1) for (_from, _to) in idxs]
            img = np.concatenate(img, axis=0).astype(np.uint8)

            masks_arr = []
            for bg_idx in range(num_bgs):
                mask = masks[bg_idx]
                temp = [mask if idx==bg_idx else np.zeros_like(masks[idx]) for idx in range(num_bgs)]
                mask = [np.concatenate(temp[_from:_to], axis=1) for (_from, _to) in idxs]
                mask = np.concatenate(mask, axis=0).astype(np.int32)
                masks_arr.append(mask)
            masks = masks_arr

        mask = masks[target_idx]
        mask = mask.astype(np.uint8)
        # mask[mask>0] = 1

        img_size = img.shape[:2]
        mat, mat_inv = self.getTransformMat(img_size, True)
        img = cv2.warpAffine(
            img,
            mat,
            self.input_size,
            flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])

        sents = sents_arr[target_idx]



        if self.mode == 'train':
            mask = cv2.warpAffine(mask,
                                mat,
                                self.input_size,
                                flags=cv2.INTER_LINEAR,
                                borderValue=0.)

            caption_raw = ' '.join(sent.replace(',', ' ,').split())
            word_vec, char_to_token_map_list_raw = tokenize_en(caption_raw, self.word_length, True)
            word_vec = word_vec.squeeze(0)

            if self.if_two_branches:
                sent_new = sents_new[target_sent_idx_en]
                caption_new = ' '.join(sent_new.replace(',', ' ,').split())
                word_vec_new, char_to_token_map_list = tokenize_en(caption_new, self.word_en_length, True)
                word_vec_new = word_vec_new.squeeze(0)
                combined_word_vec = torch.cat([word_vec, word_vec_new], dim=0)
            else:
                combined_word_vec = word_vec

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

                if self.if_two_branches:
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
                    return img, combined_word_vec, mask, en_labels_raw
            else:
                img, mask = self.convert(img, mask)
                return img, combined_word_vec, mask

        seg_id = seg_ids[target_idx]
        mask_dir = os.path.join(self.mask_dir, str(seg_id) + '.png')
        if self.mode == 'val':
            sent = sents[0]
            caption_raw = ' '.join(sent.replace(',', ' ,').split())
            word_vec, char_to_token_map_list_raw = tokenize_en(caption_raw, self.word_length, True)
            word_vec = word_vec.squeeze(0)

            if self.if_two_branches:
                sent_new = sents_new[0]
                caption_new = ' '.join(sent_new.replace(',', ' ,').split())
                word_vec_new, char_to_token_map_list = tokenize_en(caption_new, self.word_en_length, True)
                word_vec_new = word_vec_new.squeeze(0)
                combined_word_vec = torch.cat([word_vec, word_vec_new], dim=0)
            else:
                combined_word_vec = word_vec

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
                if self.if_two_branches:
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
                    assert len(org_img_sizes) == 1
                    params = {
                        'mask_dir': mask_dir,
                        'inverse': mat_inv,
                        'ori_size': np.array(org_img_sizes[0])
                    }
                    return img, combined_word_vec,mask, params, en_labels, en_labels_raw
                else:
                    img = self.convert(img)[0]
                    assert len(org_img_sizes) == 1
                    params = {
                        'mask_dir': mask_dir,
                        'inverse': mat_inv,
                        'ori_size': np.array(org_img_sizes[0])
                    }
                    return img, combined_word_vec, mask, params, en_labels_raw
            else:
                img = self.convert(img)[0]
                assert len(org_img_sizes) == 1
                params = {
                    'mask_dir': mask_dir,
                    'inverse': mat_inv,
                    'ori_size': np.array(org_img_sizes[0])
                }
                return img, combined_word_vec, mask, params

        else:
            img = self.convert(img)[0]
            assert len(org_img_sizes) == 1
            params = {
                'ori_img': ori_img,
                'seg_id': seg_id,
                'mask_dir': mask_dir,
                'inverse': mat_inv,
                'ori_size': np.array(org_img_sizes[0]),
                'sents': sents
            }
            is_decouple = self.is_decouple
            return img, mask, params, is_decouple


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
    def save_debug_images(self, img_tensor, mask_tensor, save_dir, name_prefix="sample"):
        os.makedirs(save_dir, exist_ok=True)

        # 反归一化处理（你自己设定的 mean 和 std）
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        img_denorm = img_tensor * std + mean  # 恢复原始像素值
        img_clamped = torch.clamp(img_denorm, 0, 1)  # 防止溢出
        save_image(img_clamped, os.path.join(save_dir, f"{name_prefix}_img.jpg"))

        # 处理 mask：假设是单通道、float 类型（可以是 0-1 或 0-255）
        if mask_tensor is not None:
            mask_np = mask_tensor.numpy()
            mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
            mask_img.save(os.path.join(save_dir, f"{name_prefix}_mask.png"))