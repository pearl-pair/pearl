"""***********************************************************************
Entity-Aware Generative Retrieval for Personalized Contexts

-------------------------------------------------------------------------
File: run.py
- The main python file works with multiple command-line arguments in CLI.

Version: 1.0
***********************************************************************"""


import os
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from functools import partial
from data import IndexingTrainDataset, GenerateDataset, IndexingCollator, QueryEvalCollator
from transformers import (
    T5Tokenizer,
    T5TokenizerFast,
    T5ForConditionalGeneration,
    TrainingArguments,
    MT5Tokenizer,
    MT5TokenizerFast,
    MT5ForConditionalGeneration,
    HfArgumentParser,
    set_seed,
)
from trainer import PearlTrainer, DocTqueryTrainer, build_trie, get_valid_next_tokens
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
import argparse
import random
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

def fix_seed(seed: int):
    # fix python random seed
    random.seed(seed)
    # fix numpy seed
    np.random.seed(seed)
    # fix torch seed
    torch.manual_seed(seed)
    # fix CUDA seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # fix CuDNN seed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(seed)


@dataclass
class RunArguments:
    model_name: str = field(default=None)
    model_path: Optional[str] = field(default=None)
    max_length: Optional[int] = field(default=32)
    id_max_length: Optional[int] = field(default=20)
    remove_prompt: Optional[bool] = field(default=False)
    train_file: str = field(default=None)
    valid_file: str = field(default=None)
    task: str = field(default=None, metadata={"help": "PEARL, docTquery, generation"})
    top_k: Optional[int] = field(default=10)
    num_return_sequences: Optional[int] = field(default=5)
    q_max_length: Optional[int] = field(default=64)


def add_arguments():
    parser_ = argparse.ArgumentParser(description='PEARL')
    parser_.add_argument('--from_yaml', type=str, default='configs/train.yml',
                         help='override the config from the indicated yaml')
    # Explicit local_rank from os for `torch distirbuted running as module`
    parser_.add_argument('--local_rank', type=str, default=os.environ['LOCAL_RANK'])
    return parser_


def is_between(check_time: str, start_time: str, end_time: str) -> bool:
    return start_time <= check_time <= end_time


def make_compute_metrics(cache_outputs, tokenizer, reranker=None, reranker_tokenizer=None,
                         passage_dataset=None, query_dataset=None, passage_time_mapping=None, query_time_mapping=None,
                         args_=None):
    def compute_metrics(eval_preds):
        hit_at_1 = 0
        hit_at_5 = 0
        hit_at_10 = 0
        mrr_at_10 = 0
        hit_at_50 = 0
        hit_at_100 = 0
        all_filtered_rank_list = []
        all_rr_rank_list = []

        for pair_idx, (beams, label) in enumerate(zip(eval_preds.predictions, eval_preds.label_ids)):
            rank_list = tokenizer.batch_decode(beams, skip_special_tokens=True)
            label_id = tokenizer.decode(label, skip_special_tokens=True)
            # filter out duplicates and invalid docids
            filtered_rank_list = []  # It's list of actual id if unique mapping, and ``it's list of index if not``
            for docid in rank_list:
                if args_.unique_mapping:  # For unique mapping, use identifier as index
                    if docid not in filtered_rank_list and docid in list(passage_dataset.keys()):
                        filtered_rank_list.append(docid)
                else: # For non-unique mapping, use all index corresponds to the identifier (collision)
                    for pidx_, v_ in passage_dataset.items():
                        if v_['text_id'] == docid:  # Store all idx of passage if docid matches
                            filtered_rank_list.append(pidx_)

            time_filtered_rank_list = []
            if args_.unique_mapping:
                q_t_s_time, q_t_e_time = (query_time_mapping[label_id]['t_s_time'],
                                          query_time_mapping[label_id]['t_e_time'])
            else:
                q_t_s_time, q_t_e_time = (query_time_mapping[pair_idx]['t_s_time'],
                                          query_time_mapping[pair_idx]['t_e_time'])
            if q_t_s_time != '-':  # When query's target start time exists, apply time filtering
                for id_ in filtered_rank_list:
                    if args_.unique_mapping:  # Map with actual id
                        p_t_s_time, p_t_e_time, p_ct = (passage_time_mapping[id_]['t_s_time'],
                                                        passage_time_mapping[id_]['t_e_time'],
                                                        passage_time_mapping[id_]['ct'])
                    else:  # Map with index (which is saved in id_)
                        p_t_s_time, p_t_e_time, p_ct = (passage_time_mapping[id_]['t_s_time'],
                                                        passage_time_mapping[id_]['t_e_time'],
                                                        passage_time_mapping[id_]['ct'])
                    if p_t_s_time != '-':
                        is_fit = (is_between(check_time=p_t_s_time, start_time=q_t_s_time, end_time=q_t_e_time) or
                                  is_between(check_time=p_t_e_time, start_time=q_t_s_time, end_time=q_t_e_time) or
                                  is_between(check_time=p_ct, start_time=q_t_s_time, end_time=q_t_e_time))
                    else:
                        is_fit = is_between(check_time=p_ct, start_time=q_t_s_time, end_time=q_t_e_time)

                    if is_fit:
                        time_filtered_rank_list.append(id_)

                filtered_rank_list = time_filtered_rank_list

            if len(filtered_rank_list) == 0:  # If due to time filtering or docid filtering no remains, append 'X'
                all_filtered_rank_list.append(['X'])
                all_rr_rank_list.append(['X'])
            else:
                all_filtered_rank_list.append(filtered_rank_list)

            if len(filtered_rank_list) > 0:
                retrieved_passages = [passage_dataset[id_] for id_ in filtered_rank_list]
                if args_.unique_mapping:
                    query = query_dataset[label_id]
                    pairs = [[query, text_] for text_ in retrieved_passages]
                else:
                    query = query_dataset[pair_idx]['text']
                    pairs = [[query, dict_['text']] for dict_ in retrieved_passages]

                with torch.no_grad():
                    inputs = reranker_tokenizer(pairs,
                                                padding="longest",
                                                truncation=True, return_tensors='pt',
                                                ).to(reranker.device)
                    scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
                    ranks = torch.argsort(scores, descending=True)  # 1d tensor

                    # Rearrange according to the reranker scores
                    filtered_rank_list = np.array(filtered_rank_list)[ranks.detach().cpu().numpy()].tolist()
                    all_rr_rank_list.append(filtered_rank_list)

            if args_.unique_mapping:
                hits = np.where(np.array(filtered_rank_list)[:100] == label_id)[0]
            else:
                hits = np.where(np.array(filtered_rank_list)[:100] == pair_idx)[0]
            if len(hits) != 0:
                hit_at_100 += 1
                if hits[0] == 0:
                    hit_at_1 += 1
                if hits[0] < 5:
                    hit_at_5 += 1
                if hits[0] < 10:
                    hit_at_10 += 1
                    mrr_at_10 += 1 / (hits[0] + 1)
                else:
                    mrr_at_10 += 0

                if hits[0] < 50:
                    hit_at_50 += 1
        cache_outputs['all_rr'] = all_rr_rank_list
        cache_outputs['all_or'] = all_filtered_rank_list
        return {"Hits@1": hit_at_1 / len(eval_preds.predictions),
                "Hits@5": hit_at_5 / len(eval_preds.predictions),
                "Hits@10": hit_at_10 / len(eval_preds.predictions),
                "MRR@10": mrr_at_10 / len(eval_preds.predictions),
                "Hits@50": hit_at_50 / len(eval_preds.predictions),
                "Hits@100": hit_at_100 / len(eval_preds.predictions)
                }

    return compute_metrics


def conf_from_yaml(yaml_dir: str):
    dict_ = {}
    for k, v in yaml.safe_load(Path(yaml_dir).read_text()).items():
        if isinstance(v, dict):
            for k_, v_ in v.items():
                dict_[k_] = v_
        else:
            dict_[k] = v
    return dict_


def main():
    parser_ = add_arguments()
    args_, unknown = parser_.parse_known_args()
    fix_seed(313)

    # Convert unknown arguments into a dictionary
    unknown_dict = {}
    for i in range(0, len(unknown), 2):
        if i + 1 < len(unknown):
            unknown_dict[unknown[i].lstrip('-')] = unknown[i + 1]

    # Combine known and unknown arguments
    all_args = vars(args_)  # Convert to dictionary
    all_args.update(unknown_dict)
    args_ = argparse.Namespace(**all_args)
    args_.ddp_find_unused_parameters = False

    if args_.from_yaml is not None and args_.from_yaml != "None":
        yaml_config: dict = conf_from_yaml(args_.from_yaml)
        for k, v in yaml_config.items():
            setattr(args_, k, v[0])
    args_.output_dir = 'models/' + args_.run_name
    args = []
    for k_, v_ in vars(args_).items():
        args.extend(['--' + k_, str(v_)])

    parser = HfArgumentParser((TrainingArguments, RunArguments))
    training_args, run_args, remaining_args = parser.parse_args_into_dataclasses(args=args,
                                                                                 return_remaining_strings=True)

    if 'mt5' in run_args.model_name:
        tokenizer = MT5Tokenizer.from_pretrained(run_args.model_name, cache_dir='cache')
        fast_tokenizer = MT5TokenizerFast.from_pretrained(run_args.model_name, cache_dir='cache')
        if run_args.model_path:
            model = MT5ForConditionalGeneration.from_pretrained(run_args.model_path, cache_dir='cache')
        else:
            model = MT5ForConditionalGeneration.from_pretrained(run_args.model_name, cache_dir='cache')
    else:
        tokenizer = T5Tokenizer.from_pretrained(run_args.model_name, cache_dir='cache')
        fast_tokenizer = T5TokenizerFast.from_pretrained(run_args.model_name, cache_dir='cache')
        if run_args.model_path:
            model = T5ForConditionalGeneration.from_pretrained(run_args.model_path, cache_dir='cache')
        else:
            model = T5ForConditionalGeneration.from_pretrained(run_args.model_name, cache_dir='cache')

    entity_tokens = ["<PER>", "</PER>", "<LOC>", "</LOC>", "<TIM>", "</TIM>", "<EVT>", "</EVT>"]
    tokenizer.add_tokens(entity_tokens)
    fast_tokenizer.add_tokens(entity_tokens)
    model.resize_token_embeddings(len(tokenizer))

    if run_args.task == "docTquery":
        train_dataset = IndexingTrainDataset(path_to_data=run_args.train_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             tokenizer=tokenizer)

        valid_dataset = IndexingTrainDataset(path_to_data=run_args.valid_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             remove_prompt=run_args.remove_prompt,
                                             tokenizer=tokenizer)
        trainer = DocTqueryTrainer(
            do_generation=False,
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=IndexingCollator(
                tokenizer,
                padding='longest',
            ),
        )
        trainer.train()

    elif run_args.task == "PEARL":
        train_dataset = IndexingTrainDataset(path_to_data=run_args.train_file,
                                             qpa_file=args_.qpa_file if args_.use_qpa else None,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             tokenizer=tokenizer,
                                             is_train=True if args_.cl_weight else False,
                                             use_tim_mask=args_.use_tim_mask)

        def build_neg_map(leaf_ids):
            """
            leaf_ids      = [[1,2,3,4,5], [1,2,3,4,6], ...]  # tokenised IDs
            returns dict  = {(1,2,3,4,5): (1,2,3,4,6), ...}
            """
            trie = {}
            for seq in leaf_ids:
                node = trie
                for tok in seq:
                    node = node.setdefault(tok, {})
                node["_leaf"] = seq

            neg_map = {}
            for seq in leaf_ids:
                candidate = _find_sibling(seq, trie)
                if candidate is None:
                    candidate = random.choice(leaf_ids)
                    while candidate == seq:
                        candidate = random.choice(leaf_ids)
                neg_map[tuple(seq)] = tuple(candidate)
            return neg_map

        def _find_sibling(seq, trie):
            path = []
            node = trie
            for tok in seq:  # descend, store path
                path.append((tok, node))
                node = node[tok]

            for depth in range(len(path) - 1, -1, -1):
                tok, parent = path[depth]
                siblings = [k for k in parent.keys()
                            if k != "_leaf" and k != tok]
                if siblings:
                    sib_tok = random.choice(siblings)
                    sib_node = parent[sib_tok]
                    while "_leaf" not in sib_node:
                        sib_tok = random.choice([k for k in sib_node.keys()
                                                 if k != "_leaf"])
                        sib_node = sib_node[sib_tok]
                    return sib_node["_leaf"]
            return None

        neg_map = build_neg_map([id_.split(' ') for id_ in train_dataset.valid_ids])
        train_dataset.neg_map = neg_map

        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        reranker = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3', cache_dir='cache')
        reranker_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3', cache_dir='cache')
        # reranker = AutoModelForSequenceClassification.from_pretrained('BGE-reranker')  # Use ft model
        # reranker_tokenizer = AutoTokenizer.from_pretrained('BGE-reranker')
        reranker.to(training_args.device)
        reranker.eval()

        def prefix_allowed_tokens_fn_wrapper(trie):
            def prefix_allowed_tokens_fn(batch_id, input_ids):
                prefix = input_ids.tolist()
                if prefix[0] == tokenizer.pad_token_id:
                    prefix.remove(tokenizer.pad_token_id)

                next_tokens = get_valid_next_tokens(trie, prefix)
                if not next_tokens:
                    return [tokenizer.eos_token_id]
                return next_tokens

            return prefix_allowed_tokens_fn

        unseen_root_path = f"data/synthetics/syn_50k_unseen/prep" if args_.use_annotation else f"data/synthetics/syn_50k_unseen/prep_noA"
        valid_datasets = []
        passage_time_mappings = []
        query_time_mappings = []
        passage_datasets = []
        query_datasets = []
        tries = []
        unseen_files = []

        # Prepare test instances per persona
        for unseen_file in os.listdir(unseen_root_path):
            if unseen_file == 'temp' or unseen_file == 'results.csv':
                continue
            unseen_files.append(unseen_file)
            valid_dataset = IndexingTrainDataset(path_to_data=os.path.join(unseen_root_path, unseen_file, "queries_dev.json"),
                                                 max_length=run_args.max_length,
                                                 cache_dir='cache',
                                                 remove_prompt=run_args.remove_prompt,
                                                 tokenizer=tokenizer,
                                                 use_tim_mask=args_.use_tim_mask)
            valid_datasets.append(valid_dataset)

            test_cluster_token_ids = []
            for test_cluster_id in valid_dataset.valid_ids:
                test_cluster_token_ids.append(tokenizer(test_cluster_id, add_special_tokens=False)['input_ids'])
            trie = build_trie(test_cluster_token_ids)
            tries.append(trie)

            passage_dataset, query_dataset = {}, {}
            if args_.unique_mapping:
                with open(os.path.join(unseen_root_path, unseen_file, "passages_og_dev.tsv"), 'r') as f1, open(os.path.join(unseen_root_path, unseen_file, "queries_og_dev.tsv"), 'r') as f2:
                    for p_data, q_data in zip(f1, f2):
                        p_docid, passage = p_data.split('\t')
                        q_docid, query = q_data.split('\t')
                        if p_docid != q_docid:
                            raise ValueError(
                                f'the docid does not match line by line pdoc: {p_docid} and qdoc: {q_docid}')
                        passage_dataset[str(p_docid)] = passage
                        query_dataset[str(q_docid)] = query
                passage_datasets.append(passage_dataset)
                query_datasets.append(query_dataset)

                passage_time_mapping, query_time_mapping = None, None

                import datasets
                passage_time_mapping = datasets.load_dataset(
                    'json',
                    data_files=os.path.join(unseen_root_path, unseen_file, "passages_time_mapping"),
                )['train']
                query_time_mapping = datasets.load_dataset(
                    'json',
                    data_files=os.path.join(unseen_root_path, unseen_file, "queries_time_mapping"),
                )['train']
                passage_time_mapping = {
                    _['text_id']: {'ct': _['ct'].isoformat(), 't_s_time': _['t_s_time'], 't_e_time': _['t_e_time'],
                                   'text': passage_dataset[_['text_id']]}
                    for _ in passage_time_mapping}
                query_time_mapping = {
                    _['text_id']: {'ct': _['ct'].isoformat(), 't_s_time': _['t_s_time'], 't_e_time': _['t_e_time'],
                                   'text': query_dataset[_['text_id']]}
                    for _ in query_time_mapping}
                passage_time_mappings.append(passage_time_mapping)
                query_time_mappings.append(query_time_mapping)
            else:
                with open(os.path.join(unseen_root_path, unseen_file, "passages_og_dev.tsv"), 'r') as f1, open(os.path.join(unseen_root_path, unseen_file, "queries_og_dev.tsv"), 'r') as f2:
                    for idx_, (p_data, q_data) in enumerate(zip(f1, f2)):
                        p_docid, passage = p_data.split('\t')
                        q_docid, query = q_data.split('\t')
                        if p_docid != q_docid:
                            raise ValueError(
                                f'the docid does not match line by line pdoc: {p_docid} and qdoc: {q_docid}')
                        passage_dataset[idx_] = {'text_id': str(p_docid), 'text': passage}
                        query_dataset[idx_] = {'text_id': str(q_docid), 'text': query}

                passage_datasets.append(passage_dataset)
                query_datasets.append(query_dataset)

                passage_time_mapping, query_time_mapping = None, None

                import datasets
                passage_time_mapping = datasets.load_dataset(
                    'json',
                    data_files=os.path.join(unseen_root_path, unseen_file, "passages_time_mapping"),
                )['train']
                query_time_mapping = datasets.load_dataset(
                    'json',
                    data_files=os.path.join(unseen_root_path, unseen_file, "queries_time_mapping"),
                )['train']
                passage_time_mapping = {
                    idx_: {'text_id': _['text_id'], 'ct': _['ct'].isoformat(), 't_s_time': _['t_s_time'],
                           't_e_time': _['t_e_time'], 'text': passage_dataset[idx_]['text']} for idx_, _ in enumerate(passage_time_mapping)}
                query_time_mapping = {
                    idx_: {'text_id': _['text_id'], 'ct': _['ct'].isoformat(), 't_s_time': _['t_s_time'],
                           't_e_time': _['t_e_time'], 'text': query_dataset[idx_]['text']} for idx_, _ in enumerate(query_time_mapping)}
                passage_time_mappings.append(passage_time_mapping)
                query_time_mappings.append(query_time_mapping)

        ################################################################
        if args_.is_test:
            print(f"Using test model: {args_.test_model_dir}")
            model = T5ForConditionalGeneration.from_pretrained(args_.test_model_dir, cache_dir='cache')
        elif not args_.is_test and args_.continue_checkpoint_dir:
            print(f"Continue training from checkpoint dir: {args_.continue_checkpoint_dir}")
            model = T5ForConditionalGeneration.from_pretrained(args_.continue_checkpoint_dir, cache_dir='cache')
        ################################################################

        allowed_ids = tokenizer([" ".join([str(_) for _ in range(1, 21)])]).input_ids[0]
        mask = torch.ones(model.config.vocab_size, dtype=torch.bool)
        mask[allowed_ids] = False
        model.register_buffer("invalid_mask", mask)

        trainer = PearlTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_datasets,
            data_collator=IndexingCollator(
                tokenizer,
                padding='longest',
            ),
            id_max_length=run_args.id_max_length,
            compute_metrics_wrapper=partial(make_compute_metrics, tokenizer=fast_tokenizer,
                                            reranker=reranker, reranker_tokenizer=reranker_tokenizer, args_=args_),
            restrict_decode_vocab_wrapper=prefix_allowed_tokens_fn_wrapper,
            compute_metrics=None,
            restrict_decode_vocab=None,
            all_persona_data={'passage_datasets': passage_datasets,
                              'query_datasets': query_datasets,
                              'passage_time_mappings': passage_time_mappings,
                              'query_time_mappings': query_time_mappings,
                              'tries': tries},
            zero_shot_test = args_.zero_shot_test,
            fast_tokenizer = fast_tokenizer,
            cl_weight=args_.cl_weight,
            lambda_1=args_.lambda_1,
            use_tim_mask=args_.use_tim_mask,
            args_=args_
        )

        if not args_.is_test:
            trainer.train()
        else:
            all_results = trainer.predict(valid_datasets)
            all_metrics = {}
            if training_args.local_rank == 0:
                for idx_, packed_ in enumerate(zip(unseen_files, passage_datasets, query_datasets,
                                                   all_results.all_cache_outputs, all_results.all_predict_results)):
                    unseen_file, passage_dataset, query_dataset, cache_outputs, predict_results = packed_
                    passage_time_mapping, query_time_mapping = passage_time_mappings[idx_], query_time_mappings[idx_]

                    print(f"persona id: {unseen_file}\tresults: {predict_results.metrics}")
                    predict_results.metrics['num_samples'] = len(query_dataset)
                    predict_results.metrics['numHits@1'] = (predict_results.metrics['test_Hits@1'] * len(query_dataset))
                    predict_results.metrics['numHits@5'] = (predict_results.metrics['test_Hits@5'] * len(query_dataset))
                    predict_results.metrics['numHits@10'] = (predict_results.metrics['test_Hits@10'] * len(query_dataset))
                    predict_results.metrics['numHits@50'] = (predict_results.metrics['test_Hits@50'] * len(query_dataset))
                    predict_results.metrics['numHits@100'] = (predict_results.metrics['test_Hits@100'] * len(query_dataset))
                    all_metrics[unseen_file] = predict_results.metrics
                    save_path = unseen_root_path + "/{}/hits{}_{}.csv"

                    def generate_results(hits_K: int, save_path: str):
                        success_lst, fail_lst = [], []

                        for batch_pred_rr, batch_ids in tqdm(zip(cache_outputs['all_rr'], predict_results.label_ids),
                                                             desc="Writing file"):
                            true_id = fast_tokenizer.decode(batch_ids, skip_special_tokens=True)
                            query_text = query_dataset[true_id]
                            true_passage_text = passage_dataset[true_id]

                            lst = success_lst if true_id in batch_pred_rr[:hits_K] else fail_lst
                            for pred_rr_id_ in batch_pred_rr:
                                if pred_rr_id_ == 'X':
                                    lst.append({'query': query_text,
                                                'queryTime': query_time_mapping[true_id],
                                                'tPassageTime': passage_time_mapping[true_id],
                                                'tPassage': true_passage_text,
                                                'true_id': true_id,
                                                'pred_rr_id': 'X',
                                                'rrPTime': 'X',
                                                'rrP': 'X',
                                                })
                                else:
                                    lst.append({'query': query_text,
                                                'queryTime': query_time_mapping[true_id],
                                                'tPassageTime': passage_time_mapping[true_id],
                                                'tPassage': true_passage_text,
                                                'true_id': true_id,
                                                'pred_rr_id': pred_rr_id_,
                                                'rrPTime': passage_time_mapping[pred_rr_id_],
                                                'rrP': passage_dataset[pred_rr_id_],
                                                })

                        pd.DataFrame(success_lst).to_csv(save_path.format(unseen_file, hits_K, 'success'), index=False)
                        pd.DataFrame(fail_lst).to_csv(save_path.format(unseen_file, hits_K, 'fail'), index=False)

                    generate_results(hits_K=10, save_path=save_path)
                    generate_results(hits_K=50, save_path=save_path)
                    generate_results(hits_K=100, save_path=save_path)

                from datetime import datetime
                fct = datetime.now()
                fpath = os.path.join(unseen_root_path, "results.csv")
                is_exist = os.path.exists(fpath)
                (pd.DataFrame([{'pid': k, 'fct': fct, 'run_name': args_.run_name, **v} for k, v in all_metrics.items()]).
                 to_csv(fpath, index=False, mode='a' if is_exist else 'w', header=False if is_exist else True))


    elif run_args.task == 'generation':
        generate_dataset = GenerateDataset(path_to_data=run_args.valid_file,
                                           max_length=run_args.max_length,
                                           cache_dir='cache',
                                           tokenizer=tokenizer)

        trainer = DocTqueryTrainer(
            do_generation=True,
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=QueryEvalCollator(
                tokenizer,
                padding='longest',
            ),
        )
        predict_results = trainer.predict(generate_dataset,
                                          top_k=run_args.top_k,
                                          num_return_sequences=run_args.num_return_sequences,
                                          max_length=run_args.q_max_length)
        with open(f"{run_args.valid_file}.q{run_args.num_return_sequences}.docTquery", 'w') as f:
            for batch_tokens, batch_ids in tqdm(zip(predict_results.predictions, predict_results.label_ids),
                                                desc="Writing file"):
                for tokens, docid in zip(batch_tokens, batch_ids):
                    query = fast_tokenizer.decode(tokens, skip_special_tokens=True)
                    jitem = json.dumps({'text_id': generate_dataset.docid_to_char[docid.item()], 'text': query})
                    f.write(jitem + '\n')

        print('finished!')

    else:
        raise NotImplementedError("--task should be in 'PEARL' or 'docTquery' or 'generation'")


if __name__ == "__main__":
    main()
