"""
Oracle for extractive summarization models. Uses beam search to find a subset
of sentences that maximize ROUGE-1 and ROUGE-2.
"""

import argparse
import re

import numpy as np
from tqdm import tqdm

from utils import (
    flatten,
    limit_doc,
    load_elems,
    ngrams,
    write_elems,
)


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str)
parser.add_argument('--output_path', type=str)
parser.add_argument('--ext_size', type=int, default=3)
args = parser.parse_args()
print(args)
print()


def approx_rouge(hyp_ngrams, ref_ngrams):
    hyp_count = len(hyp_ngrams)
    ref_count = len(ref_ngrams)
    overlap_count = len(hyp_ngrams & ref_ngrams)
    precision = overlap_count / hyp_count if hyp_count > 0 else 0.
    recall = overlap_count / ref_count if ref_count > 0 else 0.
    f1 = 2. * (precision * recall) / (precision + recall + 1e-8)
    return f1


def greedy_search(abs_list, doc_list, budget=3):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    
    abs_tokens = _rouge_clean(' '.join(flatten(abs_list))).split()
    sents = [_rouge_clean(' '.join(sent)).split() for sent in doc_list]
    hyp_unigrams = [ngrams(sent, 1) for sent in sents]
    hyp_bigrams = [ngrams(sent, 2) for sent in sents]
    ref_unigrams = ngrams(abs_tokens, 1)
    ref_bigrams = ngrams(abs_tokens, 2)

    selected_idxs = []

    for _ in range(budget):
        curr_max_rouge = max_rouge
        curr_id = -1

        for (i, sent) in enumerate(sents):
            if i in selected_idxs:
                continue

            candidate_idxs = selected_idxs + [i]
            candidate_unigrams = set.union(
                *[set(hyp_unigrams[idx]) for idx in candidate_idxs]
            )
            candidate_bigrams = set.union(
                *[set(hyp_bigrams[idx]) for idx in candidate_idxs]
            )

            rouge1 = approx_rouge(candidate_unigrams, ref_unigrams)
            rouge2 = approx_rouge(candidate_bigrams, ref_bigrams)
            rouge_score = rouge1 + rouge2
            if rouge_score > curr_max_rouge:
                curr_max_rouge = rouge_score
                curr_id = i

        if curr_id == -1:
            return (list(sorted(selected_idxs)), max_rouge)

        selected_idxs.append(curr_id)
        max_rouge = curr_max_rouge

    return (list(sorted(selected_idxs)), max_rouge)


if __name__ == '__main__':
    hyp_list = []
    ref_list = []

    oracle_elems = []

    loader = tqdm(load_elems(args.input_path), ncols=100)
    rouge_sum = 0.

    for i, elem in enumerate(loader, 1):
        doc_list = elem['doc_list']
        abs_list = elem['abs_list']

        if sum(map(len, doc_list)) > 0:
            doc_list = [sent[:200] for sent in doc_list][:100]
            doc_list = limit_doc(doc_list)
            oracle_idxs, oracle_rouge = greedy_search(
                abs_list, doc_list, budget=args.ext_size
            )
            oracle_tokens = [doc_list[idx] for idx in oracle_idxs]
        else:
            oracle_idxs = []
            oracle_tokens = []

        hyp_list.append(oracle_tokens)
        ref_list.append([abs_list])

        rouge_sum += oracle_rouge
        loader.set_description(f'Oracle ROUGE = {rouge_sum / i * 100.:.2f}')

        oracle_elem = {
            'name': elem['name'],
            'part': elem['part'],
            'label_list': [
                1 if i in oracle_idxs else 0
                for i in range(len(elem['doc_list']))
            ],
            'scores': {
                'ROUGE': oracle_rouge,
            },
        }
        oracle_elems.append(oracle_elem)

    write_elems(oracle_elems, args.output_path)
