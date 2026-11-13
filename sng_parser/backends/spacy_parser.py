#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : spacy_parser.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/21/2018
#
# This file is part of SceneGraphParser.
# Distributed under terms of the MIT license.
# https://github.com/vacancy/SceneGraphParser
# 
# Modification: EDA
# Created: 04/30/2022
# Author: Yanmin Wu
# E-mail: wuyanminmax@gmail.com
# https://github.com/yanmin-wu/EDA 

from .. import database
from ..parser import Parser
from .backend import ParserBackend
import os

import nltk
if os.environ.get('NLTK_DATA'):
    nltk.data.path.append(os.environ['NLTK_DATA'])

from nltk.corpus import wordnet as wn
__all__ = ['SpacyParser']


def _flatten(items):
    for item in items:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        else:
            yield item

@Parser.register_backend
class SpacyParser(ParserBackend):
    """
    Scene graph parser based on spaCy.
    """

    __identifier__ = 'spacy'

    def __init__(self, model=None):
        """
        Args:
            model (str): a spec for the spaCy model. (default: en). Please refer to the
            official website of spaCy for a complete list of the available models.
            This option is useful if you are dealing with languages other than English.
        """

        try:
            import spacy
        except ImportError as e:
            raise ImportError('Spacy backend requires the spaCy library. Install spaCy via pip first.') from e

        if spacy.__version__ < '3':
            default_model = 'en'
        else:
            default_model = 'en_core_web_sm'

        self.model = model
        if self.model is None:
            self.model = default_model

        try:
            self.nlp = spacy.load(self.model)
        except OSError as e:
            raise ImportError('Unable to load the English model. Run `python -m spacy download en` first.') from e
        


    ##############################
    # BRIEF Main parsing process #
    ##############################
    def parse(self, sentence, return_doc=False):
        doc = self.nlp(sentence)

        # STEP 1. Parse entities and their modifiers, stord in [entities]
        entities = list()
        entity_chunks = list()
        for entity in doc.noun_chunks:
            ent = dict(
                span=entity.text,           # full text. eg.'a rectangular refrigerator'
                lemma_span=entity.lemma_,   # sentence lemma:'a rectangular refrigerator'
                single_head=entity.root,
                head=entity.root.text,      # root: 'refrigerator'
                head_idx=[entity.root.i],   # root index
                ent_id = None,              # node id
                head_type='None',           # object & scene
                repeat = False,             # pronoun, but not 'it'
                lemma_head=entity.root.lemma_,          # root lemma 'refrigerator'
                span_bounds=(entity.start, entity.end), # span
                modifiers=[]
            )

            # step Modifiers of the entity
            for x in reversed(tuple(entity.root.children)):
                # TODO(Jiayuan Mao @ 08/21): try to determine the number.
                # 1. article: a, the
                if x.dep_ == 'det':
                    ent['modifiers'].append({'dep': x.dep_, 'span': x.text, 'idx': x.i, 'lemma_span': x.lemma_})
                # 2. numeral: one, two
                elif x.dep_ == 'nummod':    
                    ent['modifiers'].append({'dep': x.dep_, 'span': x.text, 'idx': x.i, 'lemma_span': x.lemma_})
                # 3. adjective: rectangular
                elif x.dep_ == 'amod':
                    ent['modifiers'].append({'dep': x.dep_, 'span': x.text, 'idx': x.i, 'lemma_span': x.lemma_})
                    # adjective + adjective：there is a [large] [wooden] bookshelf . 
                    for y in x.children:
                        if y.dep_ == 'amod':
                            ent['modifiers'].append({'dep': y.dep_, 'span': y.text, 'idx': y.i, 'lemma_span': y.lemma_})
                # 4. noun compound: a beige wooden [working table]
                elif x.dep_ == 'compound':
                    # TODO 1.
                    ent['head'] = x.text + ' ' + ent['head']
                    ent['lemma_head'] = x.lemma_ + ' ' + ent['lemma_head']
                    # target id
                    ent['head_idx'] = [x.i] + ent['head_idx']
                    # # TODO 2
                    # ent['modifiers'].append({'dep': x.dep_, 'span': x.text, 'idx': x.i, 'lemma_span': x.lemma_})
                    
                    # the armchair is south of the [rightmost coffee table].
                    for y in x.children:
                        if y.dep_ == 'amod' or y.dep_ == 'compound':
                            ent['modifiers'].append({'dep': y.dep_, 'span': y.text, 'idx': y.i, 'lemma_span': y.lemma_})

            entities.append(ent)
            entity_chunks.append(entity)

        # STEP 2. Filter entities to ensure they are [objects, scenes or pronouns]
        filter_entity_chunks =[]
        filter_entities = []
        
        for idx, ec in enumerate(entity_chunks):
            noun_type = self.get_noun_type(ec.root.lemma_)
            compound_tokens = [tok for tok in ec.root.lefts if tok.dep_ == "compound"]
            if noun_type == "Object":
                entities[idx]["head_type"] = "Object"
                filter_entity_chunks.append(ec)
                filter_entities.append(entities[idx])
            elif noun_type == "Scene":
                entities[idx]["head_type"] = "Scene"
                filter_entity_chunks.append(ec)
                filter_entities.append(entities[idx])      
            elif ec.root.lemma_ in ['this', 'it', 'which', 'there', 'these', 'those']:
                filter_entity_chunks.append(ec)
                entities[idx]['head_type']='Pron'
                filter_entities.append(entities[idx]) 
            
            elif compound_tokens:
                compound_phrase = ' '.join([tok.text for tok in compound_tokens] + [ec.root.text])
                compound_lemma = ' '.join([tok.lemma_ for tok in compound_tokens] + [ec.root.lemma_])
                
                ent = entities[idx]
                ent['head'] = compound_phrase
                ent['lemma_head'] = compound_lemma
                ent['head_idx'] = [tok.i for tok in compound_tokens] + ent['head_idx']
                filter_entity_chunks.append(ec)
                
                # 使用复合后的lemma再判断类型
                noun_type = self.get_noun_type(compound_tokens[-1].lemma_) or self.get_noun_type(ec.root.lemma_)
                ent['head_type'] = noun_type if noun_type != 'Unknown' else 'Object'  # 默认归为Object
                filter_entities.append(ent)                
        # if filter_entities == []:
        #     filter_entities = entities
        #     filter_entity_chunks = entity_chunks


        # for idx, ec in enumerate(entity_chunks):
        #     # 1. object name
        #     head_id_list = sorted(entities[idx]['head_idx'])
        #     if (ec.root.lemma_ in self.obj_cls) or (database.is_object_noun(ec.root.lemma_)) or \
        #        (entities[idx]['lemma_head'] in self.obj_cls) or (database.is_object_noun(entities[idx]['lemma_head'])) or \
        #        (True in [(doc[i].lemma_ in self.obj_cls) or (database.is_object_noun(doc[i].lemma_)) for i in head_id_list]):
        #         filter_entity_chunks.append(ec)
        #         entities[idx]['head_type']='Object'
        #         filter_entities.append(entities[idx])
        #     # 2. scene
        #     elif database.is_scene_noun(ec.root.lemma_):
        #         filter_entity_chunks.append(ec)
        #         entities[idx]['head_type']='Scene'
        #         filter_entities.append(entities[idx])
        #     # 3. pronoun
        #     elif ec.root.lemma_ in ['this', 'it', 'which', 'there', 'these', 'those']:
        #         filter_entity_chunks.append(ec)
        #         entities[idx]['head_type']='Pron'
        #         filter_entities.append(entities[idx])
        #     # 4. other special cases： shelving unit， counter top
        #     elif ec.root.lemma_ == 'top' and doc[head_id_list[-1] -1].text == 'counter':
        #         filter_entity_chunks.append(ec)
        #         entities[idx]['head_type']='Object'
        #         filter_entities.append(entities[idx])

        #         # Manually construct compound nouns
        #         ent = entities[idx]
        #         ent['head'] = doc[head_id_list[-1] -1].text + ' ' + ent['head']
        #         ent['lemma_head'] = doc[head_id_list[-1] -1].lemma_ + ' ' + ent['lemma_head']
        #         # target id
        #         ent['head_idx'] = [doc[head_id_list[-1] -1].i] + ent['head_idx']

        # mask
    
        token_mask = [int(t.dep_ == 'punct') for t in doc]  # 标记句子中的标点符号

        # STEP 3. determine the main entity
        main_entity = None
        main_entity_id = 0
        tmp_entity_chunks = []
        tmp_entity = []

        subject_tokens = self.get_main_subject_tokens(doc)
        for idx, ent in enumerate(filter_entities):
            # main entity
            if ent['head_type'] != 'Object':
                continue
            if not any(tok in subject_tokens for tok in ent['head_idx']):
                continue
            if any(ignore in ent['lemma_head'] for ignore in ['wall', 'floor']):
                continue
            filter_entities[idx]['ent_id'] = 0
            main_entity = filter_entities[idx]
            main_entity_id = idx
            tmp_entity_chunks.append(filter_entity_chunks[idx])
            tmp_entity.append(ent)
            break

        if main_entity is None:
            for idx, ent in enumerate(filter_entities):
                if ent['head_type'] == 'Object' and not any(ignore in ent['lemma_head'] for ignore in ['wall', 'floor']):
                    filter_entities[idx]['ent_id'] = 0
                    main_entity = filter_entities[idx]
                    main_entity_id = idx
                    tmp_entity_chunks.append(filter_entity_chunks[idx])
                    tmp_entity.append(ent)
                    break
        # if main_entity is None:
        #     ent = filter_entities[0]
        #     filter_entities[0]['ent_id'] = 0
        #     main_entity = ent
        #     main_entity_id = 0
        #     tmp_entity_chunks.append(filter_entity_chunks[0])
        #     tmp_entity.append(ent)


        for idx, entity in enumerate(filter_entity_chunks):
            # mark as occupied
            token_mask[entity.start:entity.end] = [1]*(entity.end - entity.start)
            if idx == main_entity_id:
                continue
            tmp_entity_chunks.append(entity)
            tmp_entity.append(filter_entities[idx])
            
            # if filter_entities[idx]['head_type']=='Pron' and main_entity is not None \
            #     and idx < main_entity_id:
            #     continue
            # if idx == main_entity_id and main_entity is not None:
            #     continue

            # tmp_entity_chunks.append(entity)
            # tmp_entity.append(filter_entities[idx])
        filter_entity_chunks = tmp_entity_chunks
        filter_entities = tmp_entity

        # TODO
        ent_id = 0
        for idx, entity in enumerate(filter_entity_chunks):
            node = None
            # PRP(it) 
            if(idx > 0 and filter_entities[idx]['head_type']=='Pron'):
                # id 0
                filter_entities[idx]['ent_id'] = 0
            
            # pronoun: it is a door， the [door] is ..... 
            elif main_entity is not None \
                and filter_entities[idx]['head_type']=='Object' \
                and (filter_entities[idx]['ent_id'] == None) \
                and (filter_entities[idx]['single_head'].text == main_entity['single_head'].text) \
                and (doc[filter_entities[idx]['single_head'].i - 1].text == 'the'):
                    filter_entities[idx]['ent_id'] = 0
                    filter_entities[idx]['repeat'] = True
            else:
                if filter_entities[idx]['ent_id'] == None:
                    ent_id+=1
                    filter_entities[idx]['ent_id'] = ent_id

        # STEP 4.  relationship between nodes.
        graph_dege = []
        existing_edges = set() 
        for idx_s, entity in enumerate(filter_entity_chunks):
            relation = None

            # step 1. from object to subject
            # Find the ROOT node of an entity
            root_token, root_entity, root_conj, idx_root_ent = self.__find_root(entity, filter_entity_chunks)
            # the [curtain] is [hanging] on the [window] above the computer desk, by the bed.
            if (root_token != None) and (root_token.dep_ in ['ROOT', 'ccomp']):
                for idx_o, obj_ent in enumerate(filter_entity_chunks):
                    if (obj_ent.root.head == root_token) \
                        and (obj_ent.root.dep_ in ['nsubj', 'nsubjpass', 'attr']) \
                        and (entity != obj_ent):

                        # find the relationship between two entities.
                        rel, rel_idx, token_mask = self.__find_rel_by_root(root_token, entity, filter_entity_chunks, doc, token_mask)

                        relation = {
                                    'obj_idx': filter_entities[idx_o]['ent_id'],
                                    'object': filter_entities[idx_o]['head'],
                                    'relation': rel,
                                    'relation_idx': rel_idx,
                                    'sub_idx': filter_entities[idx_s]['ent_id'],
                                    'subject': filter_entities[idx_s]['head']
                                    }
                        rel_key = (relation['sub_idx'], relation['obj_idx'], relation['relation'])
                        if (relation['relation'] is not None) and (rel_key not in existing_edges):
                            graph_dege.append(relation)
                            existing_edges.add(rel_key)
            
            # step 2. Parallel sentences, analyzed by conjunctions
            # eg. it is sitting on carpet and there [is] a black [chair] to the right of [it].
            elif (root_conj != None):
                for idx_o, obj_ent in enumerate(filter_entity_chunks):
                    if (obj_ent.root.head == root_conj) \
                        and (obj_ent.root.dep_ in ['nsubj', 'nsubjpass', 'attr']) \
                        and (entity != obj_ent):

                        rel, rel_idx, token_mask = self.__find_rel_by_root(root_conj, entity, filter_entity_chunks, doc, token_mask)
                        
                        relation = {
                                    'obj_idx': filter_entities[idx_o]['ent_id'],
                                    'object': filter_entities[idx_o]['head'],
                                    'relation': rel,
                                    'relation_idx': rel_idx,
                                    'sub_idx': filter_entities[idx_s]['ent_id'],
                                    'subject': filter_entities[idx_s]['head']
                                    }
                        rel_key = (relation['sub_idx'], relation['obj_idx'], relation['relation'])
                        if (relation['relation'] is not None) and (rel_key not in existing_edges):
                            graph_dege.append(relation)
                            existing_edges.add(rel_key)

            # step 2. The sentence is not the main clause, if no ROOT is found.
            elif (root_token == None) and (root_entity != None):
                rel, rel_idx, token_mask = self.__find_rel_by_entity(root_entity, entity, filter_entity_chunks, doc, token_mask)

                relation = {
                            'obj_idx': filter_entities[idx_root_ent]['ent_id'], 
                            'object': filter_entities[idx_root_ent]['head'],    
                            'relation': rel,
                            'relation_idx': rel_idx,
                            'sub_idx': filter_entities[idx_s]['ent_id'],
                            'subject': filter_entities[idx_s]['head']
                            }
                rel_key = (relation['sub_idx'], relation['obj_idx'], relation['relation'])
                if (relation['relation'] is not None) and (rel_key not in existing_edges):
                    graph_dege.append(relation)
                    existing_edges.add(rel_key)

            # step 3. Beginning with a non-predicate, the omitted subject is the main entity
            # eg. there is a black wooden steel chair. [placed] between other chairs.
            if (root_token != None) and (relation == None):
                if (root_token.i > 0) and (doc[root_token.i - 1].head != root_token) and (root_token.tag_ == 'VBN'):
                    rel, rel_idx, token_mask = self.__find_rel_by_root(root_token, entity, filter_entity_chunks, doc, token_mask)
                    relation = {
                            'obj_idx': 0,
                            'object': filter_entities[0]['head'],
                            'relation': rel,
                            'relation_idx': rel_idx,
                            'sub_idx': filter_entities[idx_s]['ent_id'],
                            'subject': filter_entities[idx_s]['head']
                            }
                    rel_key = (relation['sub_idx'], relation['obj_idx'], relation['relation'])
                    if (relation['relation'] is not None) and (rel_key not in existing_edges):
                        graph_dege.append(relation)
                        existing_edges.add(rel_key)


        # STEP 5. Look for other modifiers.
        for idx, entity in enumerate(filter_entity_chunks):
            # eg: this curtain is [ridged]. it is grooved.
            i_list = []
            if entity.root.head.dep_ == 'ROOT':
                root_token = entity.root.head
                i_flag = []
                i_list = find_modify(root_token.i, token_mask, doc, i_list, i_flag)

            # eg: this is a chair with [arm].
            if (entity.root.i < len(doc)-1):
                if (doc[entity.root.i+1].dep_ == 'prep') and (doc[entity.root.i+1].head == entity.root):
                    root_token = doc[entity.root.i+1]
                    i_list = []
                    i_flag = []
                    i_list = find_modify(root_token.i, token_mask, doc, i_list, i_flag)

            if len(i_list)>1:
                for i in i_list:
                    x = doc[i]
                    filter_entities[idx]['modifiers'].append({'dep': x.dep_, 'span': x.text, 'idx': x.i, 'lemma_span': x.lemma_})\

        # STEP 6. position label(map)
        nodes = []
        for idx, ent in enumerate(filter_entities):
            if (ent['head_type'] != 'Pron') and (ent['repeat'] == False):
                target_mask = [0] * len(token_mask)
                mod_mask    = [0] * len(token_mask)
                pron_mask   = [0] * len(token_mask)
                target_char_span = []
                mod_char_span = []
                pron_char_span = []

                for id in ent['head_idx']:
                    # mask
                    target_mask[id] = 1
                    # span
                    target_char_span.append(find_char_span_by_token_idx(id, doc))

                mod_text = []
                for mod in ent['modifiers']:
                    if doc[mod['idx']].tag_ in ['VBZ', 'DT', 'CC']:
                        continue
                    # mask
                    mod_mask[mod['idx']] = 1
                    mod_text.append(mod['span'])
                    # span
                    mod_char_span.append(find_char_span_by_token_idx(mod['idx'], doc))
                
                node = {
                    'node_id': ent['ent_id'],       # node id
                    'target' : ent['head'],         # object name
                    'lemma_head': ent['lemma_head'],# lemma
                    'node_type': ent['head_type'],
                    'target_mask':target_mask,
                    'mod_text': mod_text,
                    'mod_mask': mod_mask,
                    'pron_mask': pron_mask,
                    'target_char_span': target_char_span,
                    'mod_char_span': mod_char_span,
                    'pron_char_span': pron_char_span
                }

                nodes.append(node)
            
            # pronoun
            elif main_entity is not None and idx > 0 and len(nodes) > 0 and ent['ent_id'] == 0 and \
                 (ent['head_type'] == 'Pron' or (ent['repeat'] == True)):
                pron_mask = nodes[0]['pron_mask']

                for id in ent['head_idx']:
                    pron_mask[id] = 1
                    # pron span
                    nodes[0]['pron_char_span'].append(find_char_span_by_token_idx(id, doc))

                for mod in ent['modifiers']:
                    if doc[mod['idx']].tag_ in ['VBZ', 'DT', 'CC']:
                        continue
                    # mask & text
                    nodes[0]['mod_mask'][mod['idx']] = 1
                    nodes[0]['mod_text'].append(mod['span'])
                    # span
                    nodes[0]['mod_char_span'].append(find_char_span_by_token_idx(mod['idx'], doc))

                nodes[0]['pron_mask'] = pron_mask

        # TODO special case
        if len(nodes) == 0 or nodes[0]['node_id'] != 0:
            node_mains =  get_main_subjects(doc)
            target_masks = []
            target_mask = [0] * len(token_mask)
            mod_mask    = [0] * len(token_mask)
            pron_mask   = [0] * len(token_mask)
            mod_char_span = []
            pron_char_span = []
            subject = []
            target_char_span = []
            for index, node_main in enumerate(node_mains):
                subject.append(node_main['subject'])
                target_char_span.append(node_main['range'])
                target_mask[node_main['word_index']] = 1 
                target_masks.append(target_mask)
            node = {
                'node_id': 0,
                'target' : subject,
                'lemma_head': subject,
                'node_type' : 'Object',
                'target_mask': target_masks,
                'mod_text': [],
                'mod_mask': mod_mask,
                'pron_mask': pron_mask,
                'target_char_span': target_char_span,
                'mod_char_span': mod_char_span,
                'pron_char_span': pron_char_span,

            }
            nodes.insert(0, node)

        # note position label of relationship
        for idx, node in enumerate(nodes):
            node['rel_char_span'] = []

            if idx == 0 and node['node_id'] == 0:
                rel_id = []
                for edge in graph_dege:
                    if edge['obj_idx'] == 0:
                        rel_id.append(edge['relation_idx'])

                rel_id = list(_flatten(rel_id))

                for id in rel_id:
                    node['rel_char_span'].append(find_char_span_by_token_idx(id, doc))


        return nodes, graph_dege

    @staticmethod
    def get_main_subject_tokens(doc):
        subjects = []
        for tok in doc:
            if tok.dep_ in ['nsubj', 'nsubjpass']:
                # 向上追溯依存树，判断是不是主句的谓语动词
                head = tok.head
                while head.dep_ in ['relcl', 'advcl', 'ccomp', 'xcomp']:  # 从句关系
                    head = head.head
                if head.dep_ == 'ROOT':
                    subjects.append(tok.i)
        if subjects == []:

            root = next((tok for tok in doc if tok.dep_ == "ROOT"), None)

            # Step 1: 找 nsubj / nsubjpass
            if root:
                for child in root.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        return [child]

                # Step 2: 找直接依附 ROOT 的名词
                noun_like = [child for child in root.children if child.pos_ in ["NOUN", "PROPN", "PRON"]]
                if noun_like:
                    return [noun_like[0]]

            # Step 3: 找所有名词，选离 ROOT 最近的（如果有 ROOT）
            all_nouns = [tok for tok in doc if tok.pos_ in ["NOUN", "PROPN", "PRON"]]
            if root and all_nouns:
                closest = min(all_nouns, key=lambda x: abs(x.i - root.i))
                return [closest]

            # Step 4: 找动词或形容词的名词
            for tok in doc:
                if tok.pos_ in ["ADJ", "VERB"]:
                    for child in tok.children:
                        if child.pos_ in ["NOUN", "PROPN"]:
                            return [child]

            # Step 5: 实在没有名词就返回第一个词
            return [doc[0]]
        
        return subjects
    
    @staticmethod

    def get_noun_type(lemma):
        # 首先检查 lemma 是否为空或不是标准名词
        if not lemma or not lemma.isalpha():
            return 'Unknown'

        synsets = wn.synsets(lemma, pos=wn.NOUN)

        if not synsets:
            return 'Unknown'  # 没有对应的名词词义

        for syn in synsets:
            hypernyms = syn.hypernyms()

            # 如果它是 location/place/region/structure 类的，就判定为 Scene
            if any(h.name().startswith(('location', 'area', 'place', 'region', 'structure')) for h in hypernyms):
                return 'Scene'

        # 如果词义多为行为、状态类，过滤掉
        bad_prefixes = ['position', 'state', 'condition', 'act', 'activity', 'event']
        if all(any(h.name().startswith(tuple(bad_prefixes)) for h in s.hypernyms())
                for s in synsets):
            return 'Unknown'

        return 'Object' 
    
    @staticmethod
    def __locate_noun(chunks, i):
        for j, c in enumerate(chunks):
            if c.start <= i < c.end:
                return j
        return None

    #########################################
    # BRIEF Find the ROOT node of an entity #
    #########################################
    @staticmethod
    def __find_root(entity, filter_entity_chunks):
        root_token = None
        root_entity = None
        root_conj = None
        idx = None

        tmp = entity.root
        ent_i_list = [e.root.i for e in filter_entity_chunks]  # idx of the entity

        # 1. 
        # eg: there is a small brown table on the left side of the cabinet and a smaller table on the right side of the cabinet
        if entity.root.dep_ == 'conj' and entity.root.head.i not in ent_i_list:
            return root_token, root_entity, root_conj, idx

        # find the ROOT
        for i in range(10):
            # 2. object + object complement
            # eg. the curtain is hanging on the [window] above the computer [desk]
            if(tmp.head.i in ent_i_list) and (tmp.dep_ != 'conj'):
                idx = ent_i_list.index(tmp.head.i)
                root_entity = tmp.head
                return root_token, root_entity, root_conj, idx

            # 3. eg. it is sitting on carpet and there [is] a black chair to the right of it.
            if tmp.head.dep_ == 'conj' and tmp.head.tag_ in ['VBZ']:
                idx = tmp.head.i
                root_conj = tmp.head
                return root_token, root_entity, root_conj, idx

            # 4. 
            # eg: it will be on the left , there is a small brown table on the left side of the cabinet.
            # eg: the door is located to the right of the fireplace , there is a small shelf with items on it to the right of the door .
            if tmp.dep_ == 'ccomp' and tmp.head.dep_ == 'ROOT':
                root_token = tmp
                return root_token, root_entity, root_conj, idx

            # continue
            if(tmp.dep_ != 'ROOT'):
                tmp = tmp.head
            else:
                # return
                root_token = tmp
                return root_token, root_entity, root_conj, idx
        
        # None
        return root_token, root_entity, root_conj, idx
    

    ####################################################################
    # BRIEF Find the relationship between two entities by the ROOT word#
    ####################################################################
    @staticmethod
    # def __find_rel_by_root(root_token, entity, filter_entity_chunks, doc, token_mask):
    #     rel = None
    #     rel_idx = []

    #     token = entity.root.head
    #     if entity.root.dep_ == 'conj':
    #         token = token.head

    #     end = False
    #     while not end:
    #         if token == root_token:
    #             if rel is None:
    #                 rel = token.text
    #                 rel_idx = [token.i] + rel_idx
    #                 token_mask[token.i] = 1
    #             return rel, rel_idx, token_mask
    #         else:
    #             # 遇到新的动词（但不是root_token），剪断路径（防止多个动词拼接）
    #             if token.pos_ == "VERB" and token != root_token:
    #                 break

    #             if rel is None:
    #                 rel = token.text
    #                 rel_idx = [token.i] + rel_idx
    #                 token_mask[token.i] = 1
    #             else:
    #                 rel = token.text + ' ' + rel
    #                 rel_idx = [token.i] + rel_idx
    #                 token_mask[token.i] = 1
    #                 # 查找 modifier
    #                 rel, rel_idx, token_mask = find_children(rel, rel_idx, token, doc, token_mask)

    #             token = token.head

    #     return rel, rel_idx, token_mask

    def __find_rel_by_root(root_token, entity, filter_entity_chunks, doc, token_mask):
        rel = None
        rel_idx = []

        token = entity.root.head
        # eg. it is in front of a tan table and a bike.
        # where 'bike' is 'conj' whose head is 'table', the 'table' needs to be skipped
        # otherwise 'rel' will be 'it + in front of table + bike'
        if(entity.root.dep_ == 'conj'):
            token = token.head

        end = False
        while(not end):
            if(token == root_token):
                # if (rel == None) and (token.tag_ not in ['VBN','VBZ','VBG']):
                if rel == None:     
                    rel = token.text
                    rel_idx = [token.i] + rel_idx
                    token_mask[token.i] = 1   
                elif (token.tag_ in ['VBN','VBZ','VBG']):
                    rel = rel
                    token_mask[token.i] = 1
                else:
                    rel = token.text + ' ' + rel
                    rel_idx = [token.i] + rel_idx
                    token_mask[token.i] = 1

                return rel, rel_idx, token_mask
            # continue
            else:
                if rel == None:
                    rel = token.text
                    rel_idx = [token.i] + rel_idx
                    token_mask[token.i] = 1
                elif (token.tag_ in ['VBN','VBZ','VBG']):
                    rel = rel
                else:
                    rel = token.text + ' ' + rel
                    rel_idx = [token.i] + rel_idx
                    token_mask[token.i] = 1
                    # find its modifiers
                    rel, rel_idx, token_mask = find_children(rel, rel_idx, token, doc, token_mask)

                token = token.head

        return rel, rel_idx, token_mask
    
    #########################################################################
    # BRIEF Find the relationship between two entities in a non-main clause #
    #########################################################################
    @staticmethod
    def __find_rel_by_entity(root_entity, sub_entity, filter_entity_chunks, doc, token_mask):
        rel = None
        rel_idx = []

        iter_token = sub_entity.root.head

        end = False
        while(not end):
            if(iter_token == root_entity):
                return rel, rel_idx, token_mask
            else:
                if rel == None:
                    rel = iter_token.text
                    rel_idx = [iter_token.i] + rel_idx
                    token_mask[iter_token.i] = 1
                elif (iter_token.tag_ in ['VBN','VBZ','VBG']): # delete 'is'
                    rel = rel
                else:
                    rel = iter_token.text + ' ' + rel
                    rel_idx = [iter_token.i] + rel_idx
                    token_mask[iter_token.i] = 1
                    # find its modifiers
                    rel, rel_idx, token_mask = find_children(rel, rel_idx, iter_token, doc, token_mask)
                
                iter_token = iter_token.head
        
        return rel, rel_idx, token_mask


def get_main_subjects(doc):
    subjects = []
    for tok in doc:
        if tok.dep_ in ['nsubj', 'nsubjpass']:
            # 向上追溯依存树，判断是不是主句的谓语动词
            head = tok.head
            while head.dep_ in ['relcl', 'advcl', 'ccomp', 'xcomp']:  # 从句关系
                head = head.head
            if head.dep_ == 'ROOT':
                subjects.append(tok)

    if not subjects:
        root = next((tok for tok in doc if tok.dep_ == "ROOT"), None)

        # Step 1: 找 nsubj / nsubjpass
        if root:
            for child in root.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    return [{"subject": child.text, "range": [child.idx, child.idx + len(child.text)], "word_index": child.i}]

            # Step 2: 找直接依附 ROOT 的名词
            noun_like = [child for child in root.children if child.pos_ in ["NOUN", "PROPN", "PRON"]]
            if noun_like:
                return [{"subject": noun_like[0].text, "range": [noun_like[0].idx, noun_like[0].idx + len(noun_like[0].text)], "word_index": noun_like[0].i}]

        # Step 3: 找所有名词，选离 ROOT 最近的（如果有 ROOT）
        all_nouns = [tok for tok in doc if tok.pos_ in ["NOUN", "PROPN", "PRON"]]
        if root and all_nouns:
            closest = min(all_nouns, key=lambda x: abs(x.i - root.i))
            return [{"subject": closest.text, "range": [closest.idx, closest.idx + len(closest.text)], "word_index": closest.i}]

        # Step 4: 找动词或形容词的名词
        for tok in doc:
            if tok.pos_ in ["ADJ", "VERB"]:
                for child in tok.children:
                    if child.pos_ in ["NOUN", "PROPN"]:
                        return [{"subject": child.text, "range": [child.idx, child.idx + len(child.text)], "word_index": child.i}]

        # Step 5: 实在没有名词就返回第一个词
        return [{"subject": doc[0].text, "range": [doc[0].idx, doc[0].idx + len(doc[0].text)], "word_index": doc[0].i}]
    
    return [{"subject": tok.text, "range": [tok.idx, tok.idx + len(tok.text)], "word_index": tok.i} for tok in subjects]
# subject = get_main_subject_tokens(doc)

###########################################################
# BRIEF find the modifiers preceding the relational words #
###########################################################
# eg. the left [side]
def find_children(rel, rel_idx, token, doc, token_mask):
    for i in range(1, 10):
        if doc[token.i - i].head == token:  
            rel = doc[token.i - i].text + ' ' + rel
            rel_idx = [token.i - i] + rel_idx
            token_mask[token.i - i] = 1
            continue
        else:
            return rel, rel_idx, token_mask
    return rel, rel_idx, token_mask

######################################
# BRIEF Find the modifier after ROOT #
######################################
# eg. the chair is [black and has a bent back].
def find_modify(first_token_i, token_mask, doc, i_list, i_flag):

    children_list = [c for c in doc[first_token_i].children]
    for child in children_list:
        if (token_mask[child.i] == 0) and (doc[child.i].tag_ not in ['EX']):
            # modify = modify + ' ' + child.text
            i_list.append(child.i)
            i_flag.append(0)
    
    for idx, i in enumerate(i_list):
        if i_flag[idx] == 0:
            i_flag[idx] = 1
            find_modify(i, token_mask, doc, i_list, i_flag)
    
    # id
    if token_mask[first_token_i] == 1:
        return sorted(i_list)
    else:
        return sorted([first_token_i]+i_list)
    

#######################################################################
# BRIEF Given the position of the token, output the span of the token #
#######################################################################
def find_char_span_by_token_idx(id, doc):
    """
    input:
        doc: a spacy-parsed Doc object
        id: the token index in the Doc
    output:
        char span: [char_start, char_end], the character span of the token in the original text

    Example:
        text = "the chair is on the table"
        token id = 1  -> "chair"
        output = [4, 9]
    """
    token = doc[id]
    char_start = token.idx
    char_end = token.idx + len(token.text)
    return [char_start, char_end]
# def find_char_span_by_token_idx(id, doc):
#     '''
#     input: 
#         doc: full text
#         id:  the word idx of the token
#     output:
#         char span: [char_star, char_end] 
    
#     id:    0    1   2  3   4   5    6   7   8   9   10  11 12 13   14  15  17      18
#     eg:   the chair is on the left side of the table , and to the left of another chair
#     span: 012345678......
#     the input idx of the word 'chair' is 1
#     the output char span is [4,9]
#     '''
#     doc_text = doc.text + ' ABCDEF'
#     token_list = doc_text.split()
    
#     token_text = doc[id].text
#     # # NOTE !!! If an exception is triggered here, it is usually due to a typo in the input text. 
#     # You can correct the input text manually or comment this line.
#     assert token_text in token_list[id]     
#     if token_text != token_list[id]:
#         return []
    
#     # new sentence
#     # eg. the chair is ...
#     # eg. chair is on ...
#     # eg. is on the ...
#     new_sen = ' '.join(token_list[id:])

#     # start
#     char_star = doc_text.find(new_sen)
#     # lenth
#     lenth_ = len(token_text)
#     # end
#     char_end = char_star + lenth_

#     return [char_star, char_end]