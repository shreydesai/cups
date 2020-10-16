import argparse
import math
import logging
import os
import random
import json

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler
from tqdm import tqdm, trange
from transformers import AdamW, AutoModel, AutoTokenizer

from utils import (
    batched_index_select,
    compute_token_metrics,
    get_oracle_path,
    load_elems,
    print_results,
)


def cuda(tensor, device):
    return tensor.to(device)


def unpack(tensor):
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.cpu().numpy().tolist()


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


def deep_assert(doc_list, doc_labels):
    assert len(doc_list) == len(doc_labels)
    for list_pack in zip(doc_list, doc_labels):
        if all(map(lambda x: isinstance(x, list), list_pack)):
            assert len(sent_list) == len(sent_labels)


class SummarizationProcessor:
    """
    Processor for summarization dataset. Sentences receive positive
    labels according to a ROUGE oracle.
    """

    def __init__(self, elems, oracle_elems):
        self.elems = elems
        self.oracle_elems = oracle_elems

    def get_samples(self):
        doc_lists = [elem['doc_list'] for elem in self.elems]
        abs_lists = [elem['abs_list'] for elem in self.elems]
        label_lists = [
            oracle_elem['label_list'] for oracle_elem in self.oracle_elems
        ]
        return (doc_lists, abs_lists, label_lists)


def encode_extraction_inputs(
    doc_list,
    doc_labels,
    tokenizer,
    max_seq_length,
    max_label_length,
    device,
    do_pad=True,
):
    """Encode extraction inputs."""
    input_ids = []
    segment_ids = []
    index_ids = []
    label_ids = []

    for i, (sent_list, sent_label) in enumerate(zip(doc_list, doc_labels), 1):
        # Embed sent as [CLS] w_1 ... w_n [SEP]. Only include label on [CLS]
        # (mask out the rest of the sent tokens with -1)
        sent_input_ids = tokenizer.encode(
            ' '.join(sent_list),
            add_special_tokens=True,
            do_lower_case=True,
            do_basic_tokenize=False,
        )

        # Don't include sent if it saturates batch
        if len(sent_input_ids) > max_seq_length:
            continue

        # Constrain batch to max sequence length 
        if len(input_ids) + len(sent_input_ids) > max_seq_length:
            break

        sent_segment_ids = [int(i % 2 == 0)] * len(sent_input_ids)
        cls_index = len(input_ids)
        cls_label = sent_label

        # Collect input, segment, span, and label ids
        input_ids.extend(sent_input_ids)
        segment_ids.extend(sent_segment_ids)
        index_ids.append(cls_index)
        label_ids.append(sent_label)

    # Constrain labels
    index_ids = index_ids[:max_label_length]
    label_ids = label_ids[:max_label_length]

    # Build attention mask
    attention_mask = [1] * len(input_ids)

    # Pad up to max sequence length
    if do_pad:
        input_padding_length = max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * input_padding_length
        segment_ids += [0] * input_padding_length
        attention_mask += [0] * input_padding_length

        label_padding_length = max_label_length - len(index_ids)
        index_ids += [0] * label_padding_length
        label_ids += [-1] * label_padding_length

        for item_list in (input_ids, segment_ids, attention_mask):
            assert len(item_list) == max_seq_length
        for item_list in (index_ids, label_ids):
            assert len(item_list) == max_label_length

    return (
        cuda(torch.tensor(input_ids), device).long(),
        cuda(torch.tensor(segment_ids), device).long(),
        cuda(torch.tensor(attention_mask), device).long(),
        cuda(torch.tensor(index_ids), device).long(),
        cuda(torch.tensor(label_ids), device).long(),
    )


class TextDataset(Dataset):
    def __init__(self, path, test_mode=False):
        elems = load_elems(path)
        if args.max_samples != -1 and not test_mode:
            elems = elems[:args.max_samples]

        oracle_path = get_oracle_path(path, 'ext')
        oracle_elems = load_elems(oracle_path)

        if args.max_samples != -1 and not test_mode:
            oracle_elems = oracle_elems[:args.max_samples]

        processor = SummarizationProcessor(elems, oracle_elems)
        self.doc_lists, self.abs_lists, self.label_lists = processor.get_samples()
        self.test_mode = test_mode

    def get_sample(self, i):
        return (self.doc_lists[i], self.abs_lists[i], self.label_lists[i])

    def __len__(self):
        assert len(self.doc_lists) == len(self.abs_lists) == len(self.label_lists)
        return len(self.doc_lists)

    def __getitem__(self, i):
        doc_list, abs_list, label_list = self.get_sample(i)
        deep_assert(doc_list, label_list)

        # Constrain sentences in doc_list
        doc_list = [
            sent_list[:200] for sent_list in doc_list
        ][:100]

        (
            input_ids,
            segment_ids,
            attention_mask,
            index_ids,
            label_ids,
        ) = encode_extraction_inputs(
            doc_list,
            label_list,
            tokenizer,
            args.max_seq_length,
            args.max_label_length,
            args.device,
            do_pad=(not self.test_mode),
        )
        
        batch = {
            'input_ids': input_ids,
            'segment_ids': segment_ids,
            'attention_masks': attention_mask,
            'index_ids': index_ids,
            'label_ids': label_ids,
        }

        return batch


class ExtractionModel(nn.Module):
    def __init__(self, model='bert-base-uncased'):
        super().__init__()

        hidden_dim = 768 if 'base' in model else 1024

        self.model = AutoModel.from_pretrained(model)
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()

    def forward(self, batch):
        embeddings = self.model(
            batch['input_ids'],
            attention_mask=batch['attention_masks'],
            token_type_ids=batch['segment_ids'],
        )[0]  # [B, L_doc, 768]

        cls_embeddings = batched_index_select(
            embeddings, batch['index_ids']
        )
        cls_embeddings *= (batch['label_ids'] != -1).float().unsqueeze(2)

        logits = self.classifier(self.dropout(cls_embeddings)) # [B, L_cls, 768]
        logits = logits.transpose(1, 2)  # [B, 2, L_cls]

        return logits


def train(args, train_dataset, eval_dataset, model):
    """Trains extraction model."""
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
                        save_model(model, args.ckpt_path)
                    
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

    test_loader = tqdm(load(test_dataset, batch_size=1, shuffle=False))

    # Sentence selection evaluation
    true_list = []
    pred_list = []

    for (i, batch) in enumerate(test_loader):
        with torch.no_grad():
            logits = model(batch)

        doc_list = test_dataset.doc_lists[i]
        abs_list = test_dataset.abs_lists[i]
        label_ids = batch['label_ids']

        # Form posterior distribution over sentences 
        probs = F.softmax(logits, 1)[0, 1, :]
        doc_labels = unpack(label_ids.masked_select(label_ids != -1))
        doc_probs = unpack(probs.masked_select(label_ids != -1))

        true_list.extend(doc_labels)
        pred_list.extend(doc_probs)

    print()
    print('*** testing ***')

    auroc, aupr, f1_list = compute_token_metrics(
        true_list, pred_list, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )
    results_dict = {
        'AUROC': auroc,
        'AUPR': aupr,
        'F1': f1_list,
    }
    print_results(results_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--max_label_length', type=int, default=50)
    parser.add_argument('--init_path', type=str)
    parser.add_argument('--ckpt_path', type=str, default='temp.pt')
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--dev_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--max_train_steps', type=int, default=10000)
    parser.add_argument('--max_eval_steps', type=int, default=1000)
    parser.add_argument('--grad_accumulation_steps', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=1.)
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_test', action='store_true', default=False)
    parser.add_argument('--suppress', action='store_true', default=False)
    args = parser.parse_args()
    print(args)

    if args.suppress:
        logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = cuda(ExtractionModel(args.model), args.device)

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
        test(test_dataset)
