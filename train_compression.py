import argparse
import collections
import math
import logging
import random
import re
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler
import numpy as np
from tqdm import tqdm, trange
from transformers import AdamW, AutoModel, AutoTokenizer

from utils import (
    batched_span_select,
    compute_token_metrics,
    compute_rouge,
    get_oracle_path,
    ngrams,
    load_elems,
    print_results,
    SelfAttentiveSpanExtractor,
)


torch.manual_seed(12)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def cuda(tensor, device):
    return tensor.to(device)


def unpack(tensor):
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.cpu().numpy()


def load(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def save_model(model, ckpt_path):
    torch.save(model.state_dict(), ckpt_path)


def adamw_params(model):
    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {'params': [
            p for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ], 'weight_decay': args.weight_decay},
        {'params': [
            p for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay)
        ], 'weight_decay': 0.0},
    ]
    return params


def remove_empty_slots(doc_lists, node_lists):
    _doc_lists = []
    _node_lists = []

    for (doc_list, node_list) in zip(doc_lists, node_lists):
        if len(doc_list) > 0 and len(node_list) > 0:
            _doc_lists.append(doc_list)
            _node_lists.append(node_list)

    return (_doc_lists, _node_lists)


class SummarizationProcessor:
    """Processor for summarization datasets."""

    def __init__(self, elems, ext_oracle_elems=None, cmp_oracle_elems=None):
        self.elems = elems
        self.ext_oracle_elems = ext_oracle_elems
        self.cmp_oracle_elems = cmp_oracle_elems

    def get_samples(self):
        abs_lists = []
        doc_lists = []
        node_lists = []

        for (elem, ext_oracle_elem, cmp_oracle_elem) in tqdm(
            zip(self.elems, self.ext_oracle_elems, self.cmp_oracle_elems),
            desc='processing summ samples',
        ):
            abs_list = elem['abs_list']

            doc_list = [
                sent_list
                for (sent_list, sent_label) in zip(
                    elem['doc_list'], ext_oracle_elem['label_list']
                )
                if sent_label == 1
            ]
            
            abs_lists.append(abs_list)
            doc_lists.append(doc_list)
            node_lists.append(cmp_oracle_elem['node_list'])

        doc_lists, node_lists = remove_empty_slots(
            doc_lists, node_lists 
        )

        return (abs_lists, doc_lists, node_lists)


def encode_compression_inputs(
    doc_list,
    node_list,
    tokenizer,
    max_seq_length,
    max_span_length,
    device,
    do_pad=True,
):
    """
    Encodes documents and sub-sentential spans. Each sentence and compression
    options available within that sentence are jointly encoded. Labels correspond
    to oracle deletion ground truths for each compression option.
    """

    input_ids = []
    span_ids = []
    segment_ids = []
    label_ids = []

    offset = 0

    for (i, sent_list) in enumerate(doc_list, 1):
        # Embed sent as [CLS] w_1, ..., w_n [SEP]. Create a mapping from
        # original indices to piece indices
        sent_input_ids = [tokenizer.cls_token_id]
        sent_start_index_map = {}
        sent_end_index_map = {}

        for (j, token) in enumerate(sent_list):
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            sent_input_ids.extend(token_ids)
            sent_start_index_map[j] = len(sent_input_ids) - len(token_ids)
            sent_end_index_map[j] = len(sent_input_ids) - 1

        sent_input_ids += [tokenizer.sep_token_id]
        sent_segment_ids = [int(i % 2 == 0)] * len(sent_input_ids)

        # Don't include sent if it saturates batch
        if len(sent_input_ids) > max_seq_length:
            continue

        # Constrain batch to max sequence length
        if len(sent_input_ids) + len(input_ids) > max_seq_length:
            break

        input_ids.extend(sent_input_ids)
        segment_ids.extend(sent_segment_ids)

        # Add spans within this sentence. Original span indices are
        # now pushed by the document offset
        for node in node_list:
            if node['sent_index'] == (i - 1):
                start_index = sent_start_index_map[node['start_index']] + offset
                end_index = sent_end_index_map[node['end_index'] - 1] + offset
                span_ids.append([start_index, end_index])
                label_ids.append(node['label'])

        # Offset consists of the sum of the length of sentence
        # pieces and special tokens ([CLS], [SEP])
        offset += len(sent_input_ids)

    span_ids = span_ids[:max_span_length]
    label_ids = label_ids[:max_span_length]

    """
    # To verify the spans are encoded correctly, uncomment this block
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    for (node, (start_index, end_index)) in zip(node_list, span_ids):
        original_text = node['text']
        encoded_text = ' '.join(tokens[start_index:(end_index + 1)])
        print(f'original = {original_text}')
        print(f'encoded = {encoded_text}')
        print()
    print(span_ids)
    import sys;sys.exit(1)
    """

    attention_mask = [1] * len(input_ids)

    if do_pad:
        input_padding_length = max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * input_padding_length
        segment_ids += [0] * input_padding_length
        attention_mask += [0] * input_padding_length

        span_padding_length = max_span_length - len(span_ids)
        span_ids += [[0, 0]] * span_padding_length
        label_ids += [-1] * span_padding_length

        for item_list in (input_ids, segment_ids, attention_mask):
            assert len(item_list) == max_seq_length

        for item_list in (span_ids, label_ids):
            assert len(item_list) == max_span_length

    return (cuda(torch.tensor(input_ids), device).long(),
            cuda(torch.tensor(segment_ids), device).long(),
            cuda(torch.tensor(attention_mask), device).long(),
            cuda(torch.tensor(span_ids), device).long(),
            cuda(torch.tensor(label_ids), device).long())


class TextDataset(Dataset):
    def __init__(self, path, test_mode=False):
        elems = load_elems(path)
        ext_oracle_elems = load_elems(get_oracle_path(path, 'ext'))
        cmp_oracle_elems = load_elems(get_oracle_path(path, 'cmp'))

        if args.max_samples != -1 and not test_mode:
            elems = elems[:args.max_samples]
            ext_oracle_elems = ext_oracle_elems[:args.max_samples]
            cmp_oracle_elems = cmp_oracle_elems[:args.max_samples]

        processor = SummarizationProcessor(
            elems, ext_oracle_elems, cmp_oracle_elems
        )

        self.abs_lists, self.doc_lists, self.node_lists = processor.get_samples()
        self.test_mode = test_mode

    def get_sample(self, i):
        return (self.doc_lists[i], self.node_lists[i])

    def __len__(self):
        return len(self.doc_lists)

    def __getitem__(self, i):
        doc_list, node_list = self.get_sample(i)

        (
            input_ids,
            segment_ids,
            attention_mask,
            span_ids,
            label_ids,
        ) = encode_compression_inputs(
            doc_list,
            node_list,
            tokenizer,
            args.max_seq_length,
            args.max_span_length,
            args.device,
        )

        batch = {
            'input_ids': input_ids,
            'segment_ids': segment_ids,
            'attention_masks': attention_mask,
            'span_ids': span_ids,
            'label_ids': label_ids,
        }

        return batch


class CompressionModel(nn.Module):
    def __init__(self, model, dropout):
        super().__init__()
        self.model = AutoModel.from_pretrained(model)
        hidden_dim = 768 if 'base' in model else 1024
        self.span_extractor = SelfAttentiveSpanExtractor(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()

    def forward(self, batch):
        embeddings = self.model(
            batch['input_ids'],
            attention_mask=batch['attention_masks'],
            token_type_ids=batch['segment_ids'],
        )[0]  # [batch_size, max_seq_len, 768]
        span_embeddings = self.span_extractor(
            embeddings, batch['span_ids']
        )  # [batch_size, max_span_len, 768]
        span_embeddings = self.dropout(span_embeddings)
        logits = self.classifier(span_embeddings).transpose(1, 2)
        return logits


def train(args, train_dataset, eval_dataset, model):
    """Trains compression model."""
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size
    )

    max_steps = args.max_train_steps
    max_epochs = (
        max_steps
        // (len(train_dataloader) // args.grad_accumulation_steps)
        + 1
    )
    print(f'training for {max_steps} steps ({max_epochs} epochs)')

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = AdamW(adamw_params(model), args.learning_rate)

    train_loss = 0
    train_steps = 0
    best_eval_loss = float('inf')
    eval_history = []
    model.zero_grad()

    train_iterator = trange(max_epochs, desc='[train] epoch', ncols=100)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, ncols=100)
        for (curr_step, batch) in enumerate(epoch_iterator, 1):
            model.train()

            logits = model(batch)
            loss = criterion(logits, batch['label_ids'])
            if args.grad_accumulation_steps > 1:
                loss /= args.grad_accumulation_steps
            train_loss += loss.item()
            loss.backward()

            if curr_step % args.grad_accumulation_steps == 0:
                if args.max_grad_norm > 0.:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                model.zero_grad()
                train_steps += 1
                
                epoch_iterator.set_description(
                    f'[train] step = {train_steps}, loss = {train_loss / train_steps:.4f}'
                )

                if (args.eval_interval != -1
                    and train_steps % args.eval_interval == 0):

                    eval_loss = evaluate(args, train_steps, eval_dataset, model)
                    eval_history.append(eval_loss < best_eval_loss)

                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        save_model(model, f'{args.ckpt_path}_{train_steps}.pt')
                    
                    if len(eval_history) > 5 and not any(eval_history[-5:]):
                        epoch_iterator.close()
                        train_iterator.close()
                        print('early stopping...')
                        return -1

            if train_steps == max_steps:
                epoch_iterator.close()
                train_iterator.close()
                return -1

    return 0


def evaluate(args, train_steps, eval_dataset, model):
    """Evalutes model on development dataset."""
    model.eval()

    eval_sampler = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size
    )

    max_steps = args.max_eval_steps

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    eval_loss = 0
    eval_steps = 0

    with torch.no_grad():
        epoch_iterator = tqdm(eval_dataloader, ncols=100)
        for batch in epoch_iterator:
            logits = model(batch)
            loss = criterion(logits, batch['label_ids'])
            eval_loss += loss.item()
            eval_steps += 1

            epoch_iterator.set_description(
                f'[eval] step = {train_steps}, loss = {eval_loss / eval_steps:.4f}'
            )

            if eval_steps == max_steps:
                epoch_iterator.close()
                return eval_loss / eval_steps

    return eval_loss / eval_steps


def test(test_dataset):
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()

    test_loader = tqdm(
        load(test_dataset, args.batch_size, shuffle=False),
        ncols=100,
    )

    true_list = []
    pred_list = []

    avgs = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch_probs = F.softmax(model(batch), 1)
            batch_labels = batch['label_ids']
            for j in range(batch_probs.size(0)):
                probs = unpack(batch_probs[j, 1])
                labels = unpack(batch_labels[j])

                _probs = []

                for (prob, label) in zip(probs, labels):
                    if label != -1:
                        true_list.append(label)
                        pred_list.append(prob)

                        _probs.append(prob)

                abs_list = test_dataset.abs_lists[i][0]
                doc_list = test_dataset.doc_lists[i][0]
                node_list = test_dataset.node_lists[i]

                pred_label_list = [0] * len(doc_list)
                
                opt_nodes = []

                for (node, prob) in zip(node_list, _probs):
                    if prob > 0.7:
                        for optnode in node_list:
                            if optnode['group'] == node['group']:
                                opt_nodes.append(optnode)
                        for kk in range(node['start_index'], node['end_index']):
                            pred_label_list[kk] = 1
                for node in optnodes:
                    for kk in range(node['start_index'], node['end_index']):
                        pred_label_list[kk] = 1

                words = [word for (word,lab) in zip(doc_list, pred_label_list) if lab == 0]

                abs_ngrams = set(ngrams(abs_list, 1))
                doc_ngrams = set(ngrams(words, 1))
                rouge = compute_rouge(abs_ngrams, doc_ngrams)
                avgs.append(rouge)

    results_dict = {
        'F1': np.mean(avgs)
    }
    return results_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model', type=str, default='google/electra-base-discriminator')
    parser.add_argument('--init_path', type=str)
    parser.add_argument('--ckpt_path', type=str, default='temp.pt')
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--dev_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--del_p', type=float, default=0.5)
    parser.add_argument('--max_train_steps', type=int, default=10000)
    parser.add_argument('--max_eval_steps', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--grad_accumulation_steps', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=1.)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--max_seq_length', type=int, default=256)
    parser.add_argument('--max_span_length', type=int, default=50)
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_test', action='store_true', default=False)
    parser.add_argument('--suppress', action='store_true', default=False)
    args = parser.parse_args()
    print(args)
    print()

    if args.suppress:
        logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = cuda(CompressionModel(args.model, args.dropout), args.device)

    if args.init_path:
        model.load_state_dict(torch.load(args.init_path))
        print(f'loading pre-trained model \'{args.init_path}\'')

    if args.train_path:
        train_dataset = TextDataset(args.train_path) 
        print(f'train samples = {len(train_dataset)}')

    if args.dev_path:
        dev_dataset = TextDataset(args.dev_path)
        print(f'dev samples = {len(dev_dataset)}')
        
    if args.test_path:
        test_dataset = TextDataset(args.test_path, test_mode=True)
        print(f'test samples = {len(test_dataset)}')

    if args.do_train:
        train(args, train_dataset, dev_dataset, model)
        if args.eval_interval == -1:
            save_model(model, args.ckpt_path)

    if args.do_test:
        results_dict = test(test_dataset)

        print()
        print('*** testing ***')

        print_results(results_dict)
