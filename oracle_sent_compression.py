import argparse
import collections
import copy
import json
import random
import re
import sys

from tqdm import tqdm

from utils import (
    build_string,
    compute_rouge,
    flatten,
    load_elems,
    ngrams,
    postprocess_doc,
    print_results,
    stopwords,
    write_elems,
)
from compression_utils import build_tree, find_compressions


def _rouge_clean(s):
    return re.sub(r'[^a-zA-Z0-9 ]', '', s)


def eval_compressions(abs_list, doc_list, doc_compressions):
    
    def _apply_compression(node, labels):
        _labels = copy.deepcopy(labels)
        for _i in range(node.start_index, node.end_index):
            assert 0 <= node.sent_index <= len(_labels)
            assert 0 <= _i <= len(_labels[node.sent_index])

            _labels[node.sent_index][_i] = 1  # Delete
        return _labels


    def _preprocess(tokens):
        return [
            _rouge_clean(token.lower()) for token in tokens
        ]
    

    def _score_compression(abs_list, doc_list, hyp_labels):
        abs_tokens = flatten(abs_list)

        doc_tokens = []
        assert len(doc_list) == len(hyp_labels)
        for (sent, mask) in zip(doc_list, hyp_labels):
            assert len(sent) == len(mask)
            for (token, label) in zip(sent, mask):
                if label == 0:
                    doc_tokens.append(token)

        doc_tokens = _preprocess(doc_tokens)
        doc_uni = set(ngrams(doc_tokens, 1))
        doc_bi = set(ngrams(doc_tokens, 2))

        rouge1 = compute_rouge(doc_uni, _abs_uni)
        rouge2 = compute_rouge(doc_bi, _abs_bi)

        return (rouge1, rouge2)


    _abs_tokens = _preprocess(flatten(abs_list))
    _abs_uni = set(ngrams(_abs_tokens, 1))
    _abs_bi = set(ngrams(_abs_tokens, 2))

    compressions_list = []
    doc_labels = [[0 for _ in range(len(sent))] for sent in doc_list]
    base_rouge1, base_rouge2 = _score_compression(abs_list, doc_list, doc_labels)
    base_rouge = base_rouge1 + base_rouge2

    for sent_compressions in doc_compressions:
        for i, node in enumerate(sent_compressions):
            if [node.sent_index, node.node_index] in compressions_list:
                continue

            hyp_labels = _apply_compression(node, doc_labels)
            mod_rouge1, mod_rouge2 = _score_compression(abs_list, doc_list, hyp_labels)
            mod_rouge = mod_rouge1 + mod_rouge2

            if mod_rouge > base_rouge:
                compressions_list.append([node.sent_index, node.node_index])

                # If a parent constituent gets deleted, then by definition, all child
                # constituents must also be deleted.
                for child_node in sent_compressions[i + 1:]:
                    if (
                        node.start_index <= child_node.start_index
                        and child_node.end_index <= node.end_index
                    ):
                        compressions_list.append([child_node.sent_index, child_node.node_index])

    return compressions_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--ext_oracle_path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    print(args)
    print()

    compression_oracle_elems = []

    input_loader = load_elems(args.input_path)
    extractor_loader = load_elems(args.ext_oracle_path)
    loader = tqdm(zip(input_loader, extractor_loader), ncols=100)
    empty_slots = 0

    for (i, (elem, oracle_elem)) in enumerate(loader, 1):
        abs_list = elem['abs_list']
        _doc_list = elem['doc_list']
        _parse_list = elem['parse_list']

        extractor_oracle_labels = oracle_elem['label_list']

        doc_list = [
            sent_list
            for (sent_list, sent_label) in zip(
                _doc_list, extractor_oracle_labels
            )
            if sent_label == 1
        ]
        parse_list = [
            parse
            for (parse, sent_label) in zip(
                _parse_list, extractor_oracle_labels
            )
            if sent_label == 1
        ]

        try:
            tree_list = [build_tree(parse) for parse in parse_list]
            doc_compressions = [find_compressions(tree) for tree in tree_list]
        except:
            empty_slots += 1
            doc_compressions = []

        for (sent_index, compressions) in enumerate(doc_compressions):
            for (node_index, node) in enumerate(compressions):
                node.sent_index = sent_index
                node.node_index = node_index

        oracle_list = eval_compressions(abs_list, doc_list, doc_compressions)
        oracle_labels = [[0 for _ in range(len(sent))] for sent in doc_list]
        for (sent_index, node_index) in oracle_list:
            node = doc_compressions[sent_index][node_index]
            for j in range(node.start_index, node.end_index):
                oracle_labels[sent_index][j] = 1  # Delete
        
        oracle_nodes = []
        for (sent_index, sent) in enumerate(doc_compressions):
            for (node_index, node) in enumerate(sent):
                oracle_nodes.append(
                    {
                        'tag': node.tag,
                        'text': node.text,
                        'start_index': node.start_index,
                        'end_index': node.end_index,
                        'sent_index': node.sent_index,
                        'node_index': node.node_index,
                        'label': int([node.sent_index, node.node_index] in oracle_list),
                    }
                )
        compression_oracle_elems.append(
            {
                'name': elem['name'],
                'part': elem['part'],
                'node_list': oracle_nodes,
                'label_list': oracle_labels,
            }
        )

    write_elems(compression_oracle_elems, args.output_path)

    print('done!')
    print(f'found {empty_slots} empty slots')
