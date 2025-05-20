"""***********************************************************************
Entity-Aware Generative Retrieval for Personalized Contexts

-------------------------------------------------------------------------
File: trainer.py
- The main training modules for conditional generation models.

Version: 1.0
***********************************************************************"""


import re
import torch
import datasets
import numpy as np
from torch import nn
from functools import partial
from torch.functional import F
from data import IndexingTrainDataset
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer import Trainer
from torch.utils.data import Dataset
from dataclasses import dataclass


def build_trie(sequences):
    trie = {}
    for seq in sequences:
        node = trie
        for token in seq:
            node = node.setdefault(token, {})
    return trie


def get_valid_next_tokens(trie, prefix):
    node = trie
    for token in prefix:
        if token in node:
            node = node[token]
        else:
            return []  # invalid prefix
    return list(node.keys())


@dataclass
class CustomPredictionOutput:
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]
    all_cache_outputs: Optional
    all_predict_results: Optional


class PearlTrainer(Trainer):
    def __init__(self, compute_metrics_wrapper, restrict_decode_vocab_wrapper, restrict_decode_vocab, id_max_length,
                 all_persona_data, zero_shot_test, fast_tokenizer, lambda_1, cl_weight, use_tim_mask, args_,
                 **kwds):
        super().__init__(**kwds)
        self.compute_metrics_wrapper = compute_metrics_wrapper
        self.restrict_decode_vocab_wrapper = restrict_decode_vocab_wrapper
        self.restrict_decode_vocab = restrict_decode_vocab
        self.id_max_length = id_max_length
        self.all_persona_data = all_persona_data
        self.zero_shot_test = zero_shot_test
        self.fast_tokenizer = fast_tokenizer
        self.restrict_decode_vocab_dict = {}
        self.lambda_1 = lambda_1
        self.cl_weight = cl_weight
        self.use_tim_mask = use_tim_mask
        self.args_ = args_
        self.all_entity_token_ids = list(self.tokenizer.added_tokens_encoder.values())

    def get_special_ent_tok_mask(self, ids):
        return torch.tensor([1 if id_.item() in self.all_entity_token_ids else 0 for ids_ in ids for id_ in ids_],
                            device=self.args.device).reshape(ids.shape)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels'],
                        output_attentions=True, return_dict=True)
        # Compute original CE Loss (w/ teacher-forcing)
        loss = outputs.loss
        emr_loss, cl_loss = 0.0, 0.0

        if self.lambda_1 > 0:
            entity_mask = inputs['entity_mask'].float()
            attn_mat = outputs.encoder_attentions[-1].mean(dim=1)
            attn_mass_entity = (attn_mat * entity_mask.unsqueeze(1)).sum(dim=(1, 2)) / attn_mat.sum(dim=(1, 2)) # [B]
            entity_token_count = entity_mask.sum(dim=1)

            epsilon = 1e-8
            per_token_mass = attn_mass_entity / (entity_token_count + epsilon)
            delta = self.args_.delta
            entity_loss = -torch.log(1 - torch.relu(per_token_mass - delta)).mean()
            emr_loss = entity_loss

        if self.cl_weight > 0:
            inputs_special_ent_mask = self.get_special_ent_tok_mask(inputs['input_ids'])

            tp_inputs = inputs['tp_inputs']
            tp_inputs_special_ent_mask = self.get_special_ent_tok_mask(tp_inputs['input_ids'])
            tp_outputs = model(input_ids=tp_inputs['input_ids'], attention_mask=tp_inputs['attention_mask'],
                               labels=inputs['labels'], output_attentions=True, return_dict=True)

            hn_inputs = inputs['hn_inputs']
            hn_inputs_special_ent_mask = self.get_special_ent_tok_mask(hn_inputs['input_ids'])
            hn_outputs = model.module.encoder(input_ids=hn_inputs['input_ids'],
                                              attention_mask=hn_inputs['attention_mask'],
                                              return_dict=True)

            def adaptive_entity_pooling(encoder_outputs, entity_mask, gamma=self.args_.gamma):
                global_embed = encoder_outputs.mean(dim=1)
                entity_embed = (encoder_outputs * entity_mask.unsqueeze(-1)).sum(dim=1)
                entity_embed = entity_embed / (entity_mask.sum(dim=1, keepdim=True) + 1e-8)

                combined_embed = gamma * entity_embed + global_embed
                return combined_embed

            in_enc_embed = adaptive_entity_pooling(outputs.encoder_last_hidden_state, inputs_special_ent_mask)
            tp_enc_embed = adaptive_entity_pooling(tp_outputs.encoder_last_hidden_state, tp_inputs_special_ent_mask)
            hn_enc_embed = adaptive_entity_pooling(hn_outputs.last_hidden_state, hn_inputs_special_ent_mask)

            def contrastive_loss(in_enc_embed, tp_enc_embed, hn_enc_embed, tau=0.05):
                logits_pos = (in_enc_embed @ tp_enc_embed.T) / tau
                logits_hn = (in_enc_embed @ hn_enc_embed.T) / tau
                logits = torch.cat([logits_pos, logits_hn], dim=1)
                targets = torch.arange(logits.size(0), device=logits.device)
                return torch.nn.functional.cross_entropy(logits, targets)

            cl_loss = contrastive_loss(in_enc_embed, tp_enc_embed, hn_enc_embed)

        if self.args_.restrict_train_tokens:
            logits = outputs.logits
            logits.masked_fill_(model.module.invalid_mask, -1e9)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                inputs["labels"].view(-1),
                ignore_index=-100
            )

        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            emr = emr_loss.detach().float().mean().item() if self.lambda_1 else 0
            cl = cl_loss.detach().float().mean().item() if self.cl_weight else 0
            ce = loss.detach().float().mean().item()
            self.log(
                {
                    'ce_loss': ce,
                    'emr_loss': emr,
                    'cl_loss': cl
                }
            )

        if return_outputs:
            return loss, [None, None]
        return loss + self.lambda_1 * emr_loss + self.cl_weight * cl_loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        inputs['labels'] = inputs['labels'].to(self.args.device)

        with torch.no_grad():
            num_samples = 100
            batch_beams = model.generate(
                inputs['input_ids'].to(self.args.device),
                max_length=20,
                num_beams=num_samples,
                prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                num_return_sequences=num_samples,
                early_stopping=True, )

            if batch_beams.shape[-1] < self.id_max_length:
                batch_beams = self._pad_tensors_to_max_len(batch_beams, self.id_max_length)

            inputs['labels'] = self._pad_tensors_to_max_len(inputs['labels'], self.id_max_length)
            batch_beams = batch_beams.reshape(inputs['input_ids'].shape[0], num_samples, -1)

        return (None, batch_beams, inputs['labels'])

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:

        prediction_output = self.predict(
            test_datasets=self.eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        if self.is_world_process_zero():
            # Compute mean of metrics across all datasets
            all_metrics = prediction_output.all_predict_results
            all_num_samples = [len(cache_outputs['all_rr']) for cache_outputs in prediction_output.all_cache_outputs]
            total_samples = sum(all_num_samples)

            mean_metrics = {}
            keys = list(all_metrics[0].metrics.keys())

            for key in keys:
                # number of hit samples per key (k=1, 10, ...)
                num_hits = [result.metrics[key] * all_num_samples[idx_] for idx_, result in enumerate(all_metrics)]
                mean_metrics[f"{key}"] = float(np.sum(num_hits) / total_samples)

            return mean_metrics

    def predict(
            self,
            test_datasets: List[Dataset],
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "test",
    ):
        if self.zero_shot_test:
            from transformers import set_seed
            set_seed(313)
            self.model.eval()
            for idx_ in range(len(test_datasets)):
                original_ids = list(self.all_persona_data['passage_datasets'][idx_].keys())
                if self.args_.unique_mapping:
                    original_texts = list(self.all_persona_data['passage_datasets'][idx_].values())
                else:
                    original_texts = [d_['text'] for d_ in list(self.all_persona_data['passage_datasets'][idx_].values())]
                text_input_ids = self.tokenizer(original_texts,
                                                return_tensors="pt",
                                                padding="longest",
                                                truncation='only_first',
                                                ).input_ids
                batch_size = 4
                num_samples = 150
                num_returns = 150
                new_ids = []
                for start in range(0, len(text_input_ids), batch_size):
                    end = start + batch_size
                    batch_new_ids = self.model.generate(text_input_ids[start:end].to(self.args.device),
                                                        max_length=20 - 1,
                                                        num_beams=num_samples,
                                                        num_return_sequences=num_returns,
                                                        early_stopping=True)
                    batch_new_ids = self.fast_tokenizer.batch_decode(batch_new_ids,
                                                                     skip_special_tokens=True,)
                    for s_idx in range(0, len(batch_new_ids), num_returns):
                        if self.args_.unique_mapping:
                            for _idx, new_id in enumerate(batch_new_ids[s_idx:s_idx + num_returns]):
                                if not re.sub(r' {2,}', ' ', new_id.strip()) in new_ids:
                                    new_ids.append(re.sub(r' {2,}', ' ', new_id.strip()))
                                    break
                        else:
                            new_ids.append(re.sub(r' {2,}', ' ', batch_new_ids[s_idx].strip()))

                assert len(new_ids) == len(original_ids), (f'Unique id mapping failed. '
                                                           f'len(original_ids) = {len(original_ids)}, '
                                                           f'len(new_ids) = {len(new_ids)}')

                ogid_to_newid = {k: v for k, v in zip(original_ids, new_ids)}
                test_datasets[idx_] = IndexingTrainDataset(load_dataset=datasets.Dataset.from_dict({
                    "text": test_datasets[idx_].train_data['text'],
                    "text_id": new_ids
                }),
                    cache_dir='cache',
                    max_length=test_datasets[idx_].max_length,
                    tokenizer=self.tokenizer,
                    remove_prompt=True,
                    dataset_index=idx_,
                    use_tim_mask=self.use_tim_mask
                )

                def map_dict_ids_to_new_ids(og_dict: dict):
                    return {ogid_to_newid[k]: v for k, v in og_dict.items()}

                def map_dict_ids_to_new_listify_ids(og_dict: dict, new_ids_: list):
                    for k_, v_ in og_dict.items():
                        v_['text_id'] = new_ids_[k_]
                    return og_dict

                mapping_func = map_dict_ids_to_new_ids if self.args_.unique_mapping \
                    else partial(map_dict_ids_to_new_listify_ids, new_ids_=new_ids)

                self.all_persona_data['passage_datasets'][idx_] = (
                    mapping_func(self.all_persona_data['passage_datasets'][idx_]))
                self.all_persona_data['query_datasets'][idx_] = (
                    mapping_func(self.all_persona_data['query_datasets'][idx_]))
                self.all_persona_data['passage_time_mappings'][idx_] = (
                    mapping_func(self.all_persona_data['passage_time_mappings'][idx_]))
                self.all_persona_data['query_time_mappings'][idx_] = (
                    mapping_func(self.all_persona_data['query_time_mappings'][idx_]))

                self.all_persona_data['tries'][idx_] = build_trie(
                    self.fast_tokenizer(new_ids, add_special_tokens=False)['input_ids'])
                self.restrict_decode_vocab_dict[idx_] = (
                    self.restrict_decode_vocab_wrapper(self.all_persona_data['tries'][idx_]))

        all_cache_outputs, all_predict_results = [], []
        for idx_, test_dataset in enumerate(test_datasets):
            cache_outputs = {}
            self.compute_metrics = self.compute_metrics_wrapper(cache_outputs=cache_outputs,
                                                                passage_dataset=
                                                                self.all_persona_data['passage_datasets'][idx_],
                                                                query_dataset=self.all_persona_data['query_datasets'][
                                                                    idx_],
                                                                passage_time_mapping=
                                                                self.all_persona_data['passage_time_mappings'][idx_],
                                                                query_time_mapping=
                                                                self.all_persona_data['query_time_mappings'][idx_])
            self.restrict_decode_vocab = self.restrict_decode_vocab_dict[idx_]
            predict_results = super().predict(test_dataset, ignore_keys=ignore_keys,
                                              metric_key_prefix=metric_key_prefix)
            all_cache_outputs.append(cache_outputs)
            all_predict_results.append(predict_results)

        res = all_predict_results[0]

        if self.is_world_process_zero():
            prediction_output = CustomPredictionOutput(
                predictions=res.predictions,
                label_ids=res.label_ids,
                metrics=res.metrics,
                all_cache_outputs=all_cache_outputs,
                all_predict_results=all_predict_results,
            )

            # Compute mean of metrics across all datasets
            all_metrics = prediction_output.all_predict_results
            all_num_samples = [len(cache_outputs['all_or']) for cache_outputs in prediction_output.all_cache_outputs]
            total_samples = sum(all_num_samples)

            mean_metrics = {}
            keys = list(all_metrics[0].metrics.keys())

            for key in keys:
                num_hits = [result.metrics[key] * all_num_samples[idx_] for idx_, result in enumerate(all_metrics)]
                mean_metrics[f"{key}"] = float(np.sum(num_hits) / total_samples)

            print(mean_metrics)
            return prediction_output

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")
        tensor[tensor == -100] = self.tokenizer.pad_token_id
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor


class DocTqueryTrainer(Trainer):
    def __init__(self, do_generation: bool, **kwds):
        super().__init__(**kwds)
        self.do_generation = do_generation

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                     labels=inputs['labels']).loss
        if return_outputs:
            return loss, [None, None]
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if not self.do_generation:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        outputs = self.model.generate(
            input_ids=inputs[0]['input_ids'].to(self.args.device),
            attention_mask=inputs[0]['attention_mask'].to(self.args.device),
            max_length=self.max_length,
            do_sample=True,
            top_k=self.top_k,
            num_return_sequences=self.num_return_sequences)
        labels = torch.tensor(inputs[1], device=self.args.device).repeat_interleave(self.num_return_sequences)

        if outputs.shape[-1] < self.max_length:
            outputs = self._pad_tensors_to_max_len(outputs, self.max_length)
        return (None, outputs.reshape(inputs[0]['input_ids'].shape[0], self.num_return_sequences, -1),
                labels.reshape(inputs[0]['input_ids'].shape[0], self.num_return_sequences, -1))

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def predict(
            self,
            test_dataset: Dataset,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "test",
            max_length: Optional[int] = None,
            num_return_sequences: Optional[int] = None,
            top_k: Optional[int] = None,
    ):

        self.max_length = max_length
        self.num_return_sequences = num_return_sequences
        self.top_k = top_k
        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
