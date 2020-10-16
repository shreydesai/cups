import argparse
import json

import numpy as np
from tqdm import tqdm

from utils import load_elems


def build_extractive_spans(abs_tokens, doc_tokens):
    span_list = []
    i = 0
    j = 0

    while i < len(abs_tokens):
        f = []
        while j < len(doc_tokens):
            if abs_tokens[i] == doc_tokens[j]:
                _i, _j = i, j
                while (
                    _i < len(abs_tokens)
                    and _j < len(doc_tokens)
                    and abs_tokens[_i] == doc_tokens[_j]
                ):
                    _i, _j = _i + 1, _j + 1
                if len(f) < (_i - i - 1):
                    f = abs_tokens[i:_i]
                j = _j
            else:
                j += 1
        i += max(len(f), 1)
        j = 0
        if f:
            span_list.append(f)

    return span_list


def extractive_coverage(abs_tokens, ext_spans):
    return (
        (1. / len(abs_tokens)) * sum(map(len, ext_spans))
        if len(abs_tokens) > 0 else 0.
    )


def extractive_density(abs_tokens, ext_spans):
    return (
        (1. / len(abs_tokens)) * sum(map(lambda x: x ** 2, map(len, ext_spans)))
        if len(abs_tokens) > 0 else 0.
    )


def compression_metrics(abs_list, doc_list):
    abs_tokens = [word for sent in abs_list for word in sent]
    doc_tokens = [word for sent in doc_list for word in sent]
    ext_spans = build_extractive_spans(abs_tokens, doc_tokens)
    coverage = extractive_coverage(abs_tokens, ext_spans)
    density = extractive_density(abs_tokens, ext_spans)
    return (coverage, density)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    args = parser.parse_args()
    
    elems = load_elems(args.path)
    coverage_list = []
    density_list = []

    for elem in tqdm(elems, ncols=100):
        coverage, density = compression_metrics(
            elem['abs_list'], elem['doc_list']
        )
        coverage_list.append(coverage)
        density_list.append(density)

    out_elems = []
    for (x,y) in zip(coverage_list, density_list):
        out_elems.append({'coverage': x, 'density': y})

    with open('curation_cmp.txt', 'w+') as f:
        for elem in out_elems:
            f.write(f'{json.dumps(elem)}\n')
    
    #print('coverage', np.median(coverage_list))
    #print('density', np.median(density_list))
    #print()
