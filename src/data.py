"""***********************************************************************
Entity-Aware Generative Retrieval for Personalized Contexts

-------------------------------------------------------------------------
File: data.py
- The dataset indexing pipeline.

Version: 1.0
***********************************************************************"""


import re
import torch
import datasets
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
TIM_CLEANER = re.compile(r"<TIM\b[^>]*?>.*?</TIM>", re.DOTALL)


class IndexingTrainDataset(Dataset):
    def __init__(
            self,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
            path_to_data: str = None,
            load_dataset = None,
            remove_prompt=False,
            dataset_index=None,
            is_train=False,
            qpa_file=None,
            use_tim_mask=False,
            neg_map = None
    ):
        if path_to_data is not None:
            self.train_data = datasets.load_dataset(
                'json',
                data_files=[path_to_data, qpa_file] if qpa_file else [path_to_data],
                cache_dir=cache_dir
            )['train']
        elif load_dataset is not None:
            self.train_data = load_dataset
        else:
            raise ValueError('Failed loading either dataset')

        self.is_train = is_train
        self.max_length = max_length
        self.use_tim_mask = use_tim_mask
        self.tokenizer = tokenizer
        self.remove_prompt = remove_prompt
        self.total_len = len(self.train_data)
        self.valid_ids = set()
        self.id_to_tp_text = {}
        self.neg_map = neg_map
        for idx, data in enumerate(self.train_data):
            self.valid_ids.add(str(data['text_id']))
            if self.is_train and data['is_tp']:
                self.id_to_tp_text[str(data['text_id'])] = TIM_CLEANER.sub("<TIM>", data['text']) \
                    if self.use_tim_mask else data['text']

        self.dataset_index = dataset_index
        self.entity_token_to_id: dict = self.tokenizer.added_tokens_encoder
        self.entity_token_ids: list = [v for k, v in self.entity_token_to_id.items()
                                       if k != '<TIM>' and k != '</TIM>']

    def get_ent_tok_mask(self, ids):
        """
        Returns: A vector for entity tokens
            - "<ENT> TOK_1 TOK_2 </ENT> TOK_3" = 0 1 1 0 0
        """
        ent_tok_mask = []
        is_ent_tok = False
        for id_ in ids:
            ent_tok_mask.append(1 if is_ent_tok else 0)
            if id_ in self.entity_token_ids:
                is_ent_tok = not is_ent_tok
                ent_tok_mask[-1] = 0
        return torch.tensor(ent_tok_mask)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        data = self.train_data[item]
        if self.remove_prompt:
            data['text'] = data['text'][9:] if data['text'].startswith('Passage: ') else data['text']
            data['text'] = data['text'][10:] if data['text'].startswith('Question: ') else data['text']

        if self.use_tim_mask:
            data['text'] = TIM_CLEANER.sub("<TIM>", data['text'])

        input_ids = self.tokenizer(data['text'],
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]
        entity_mask = self.get_ent_tok_mask(input_ids)

        tp_ids = None
        hn_ids = None
        tp_entity_mask = None
        hn_entity_mask = None
        neg_id_str = None
        if self.is_train:
            tp_ids = self.tokenizer(self.id_to_tp_text[str(data['text_id'])],
                                    return_tensors="pt",
                                    truncation='only_first',
                                    max_length=self.max_length).input_ids[0]
            tp_entity_mask = self.get_ent_tok_mask(tp_ids)
            neg_id_tokens = self.neg_map[tuple(map(str, data['text_id'].split(' ')))]
            neg_id_str = ' '.join(map(str, neg_id_tokens))
            hn_ids = self.tokenizer(self.id_to_tp_text[neg_id_str],
                                    return_tensors="pt",
                                    truncation='only_first',
                                    max_length=self.max_length).input_ids[0]
            hn_entity_mask = self.get_ent_tok_mask(hn_ids)

        return (input_ids, str(data['text_id']), self.dataset_index, entity_mask, tp_ids, tp_entity_mask, hn_ids,
                hn_entity_mask, neg_id_str)


class GenerateDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
    ):
        self.data = []
        self.docid_to_char: dict = {}
        with open(path_to_data, 'r') as f:
            for data in f:
                if 'synthetic' in path_to_data:
                    docid_char, passage = data.split('\t')
                    docid = int(''.join(docid_char.split(' ')))
                    self.docid_to_char[docid] = docid_char
                    self.data.append((docid, f'{passage}'))
                else:
                    raise NotImplementedError(f"dataset {path_to_data} for docTquery generation is not defined.")

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.total_len = len(self.data)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        docid, text = self.data[item]
        input_ids = self.tokenizer(text,
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]
        return input_ids, docid


@dataclass
class IndexingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        if features[0][4] is not None:
            tp_ids = [{'input_ids': x[4]} for x in features]
            inputs['tp_inputs'] = super().__call__(tp_ids)
            inputs['tp_inputs']['entity_mask'] = self.tokenizer.pad([{'input_ids': x[5]} for x in features])['input_ids']

        if features[0][6] is not None:
            hn_ids = [{'input_ids': x[6]} for x in features]
            inputs['hn_inputs'] = super().__call__(hn_ids)
            inputs['hn_inputs']['entity_mask'] = self.tokenizer.pad([{'input_ids': x[7]} for x in features])['input_ids']

        inputs['entity_mask'] = self.tokenizer.pad([{'input_ids': x[3]} for x in features])['input_ids']

        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids

        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        return inputs

@dataclass
class QueryEvalCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        labels = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        return inputs, labels
