import argparse
import random
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from termcolor import colored
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from compute_compression import compression_metrics
from compression_utils import build_tree, find_compressions
from train_extraction import ExtractionModel, encode_extraction_inputs
from train_compression import CompressionModel, encode_compression_inputs
from utils import (
    build_string,
    compute_approx_rouge,
    compute_perl_rouge,
    load_elems,
    ngrams,
    postprocess_doc,
    print_results,
    write_elems,
)

random.seed(1)


def _check_dims(list1, list2):
    assert len(list1) == len(list2)
    if isinstance(list1[0], list) and isinstance(list2[0], list):
        for (x, y) in zip(list1, list2):
            assert len(x) == len(y)


def cuda(tensor):
    return tensor.to(args.device)


def unpack(tensor):
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.cpu().numpy().tolist()


def trigram_overlap(x, y):
    x_tri = ngrams(x, 3)
    y_tri = ngrams(y, 3)
    return any(_y_tri in x_tri for _y_tri in y_tri)


def compute_rate(label_list):
    n_delete = 0
    n_total = 0
    for sent_list in label_list:
        for label in sent_list:
            if label == 1:
                n_delete += 1
            n_total += 1
    return (
        n_delete / n_total
        if n_total > 0 else 0.
    )


def run_extraction(args, model, tokenizer, doc_list):
    """
    Runs extraction model. Using document list, forms a posterior distribution
    over sentences, then selects the top-k sentences with the highest score.
    """

    (
        input_ids,
        segment_ids,
        attention_masks,
        index_ids,
        label_ids,
    ) = encode_extraction_inputs(
        doc_list,
        [0] * len(doc_list),
        tokenizer,
        args.max_seq_length,
        args.max_label_length,
        args.device,
        do_pad=False,
    )

    batch = {
        'input_ids': input_ids.unsqueeze(0),
        'segment_ids': segment_ids.unsqueeze(0),
        'attention_masks': attention_masks.unsqueeze(0),
        'index_ids': index_ids.unsqueeze(0),
        'label_ids': label_ids.unsqueeze(0),
    }

    with torch.no_grad():
        logits = model(batch)

    # Extract probabilities for each sentence
    probs = F.softmax(logits, 1)[0, 1, :]
    doc_probs = unpack(probs.masked_select(label_ids != -1))

    # Sort sentence probabilities, then optionally use trigram blocking to
    # remove sentences with duplicate content
    ext_list = []
    sorted_doc_indices = np.argsort(doc_probs)[::-1]
    for ext_sent_index in sorted_doc_indices:
        if len(ext_list) == args.k:
            break
        ext_sent = doc_list[ext_sent_index]
        if (
            (
                args.trigram_block and not any(
                    trigram_overlap(ext_sent, prev_ext_sent)
                    for (_, prev_ext_sent) in ext_list
                )
            )
            or (not args.trigram_block)
        ):
            # Add sentence to list respecting document order. Because we
            # only take the top-k sentences, the entire document does
            # not form the overall summary.
            ext_list.append((ext_sent_index, ext_sent))
            ext_list = list(sorted(ext_list, key=lambda x: x[0]))

    # Extract sentences and indices
    ext_sents_indices, ext_sents = list(zip(*ext_list))

    return (list(ext_sents), ext_sents_indices)


def derive_compressions(parse_list, ext_sents, ext_sents_indices):
    """Derive space of possible compressions."""

    ext_parse = [parse_list[index] for index in ext_sents_indices]
    tree_list = [build_tree(parse) for parse in ext_parse]
    doc_compressions = [
        find_compressions(tree, include_optional=True) for tree in tree_list
    ]

    compressions_list = []
    for (sent_index, sent_compressions) in enumerate(doc_compressions):
        for (node_index, node) in enumerate(sent_compressions):
            node.sent_index = sent_index
            node.node_index = node_index
            node.label = 0
            compressions_list.append(node.to_json())
    
    # Constrain space to max span length.
    compressions_list = compressions_list[:args.max_span_length]

    return compressions_list


def apply_compression(label_list, node):
    """Applies compression node on labels."""

    for i in range(node['start_index'], node['end_index']):
        label_list[node['sent_index']][i] = 1  # Delete


def run_grammar(args, model, tokenizer, parse_list, ext_sents, compressions_list):
    """
    Runs grammar model. From the extracted sentences, uses a constituency tree to
    enumerate compression options, which the model then chooses from. Resulting
    compression options are then used in the compression model.
    """

    # Split compressions list into optional and non-optional compressions. Only the
    # non-optional compressions will be used for evaluation. Among the options
    # that are selected, the optional compressions that belong to the same group
    # will also be selected post-hoc.
    eval_compressions_list = [
        node for node in compressions_list if not node['optional']
    ]

    try:
        # Encode inputs, create batch, and run grammar model.
        (
            input_ids,
            segment_ids,
            attention_mask,
            span_ids,
            label_ids,
        ) = encode_compression_inputs(
            ext_sents,
            eval_compressions_list,
            tokenizer,
            args.max_seq_length,
            args.max_span_length,
            args.device,
            do_pad=True,
        )

        batch = {
            'input_ids': input_ids.unsqueeze(0),
            'segment_ids': segment_ids.unsqueeze(0),
            'attention_masks': attention_mask.unsqueeze(0),
            'span_ids': span_ids.unsqueeze(0),
            'label_ids': label_ids.unsqueeze(0),
        }

        with torch.no_grad():
            logits = model(batch)

        # Obtain deletion probabilities for each span.
        probs = F.softmax(logits, 1)[0, 1, :]
        span_probs = unpack(probs.masked_select(label_ids != -1))
    except:
        span_probs = []

    grm_compressions_list = []

    if len(span_probs) > 0:
        assert len(eval_compressions_list) == len(span_probs)

        # Build up list of grammatical compressions.
        for (node, prob) in zip(eval_compressions_list, span_probs):
            # If node meets grammar threshold, select it. Also find optional
            # nodes that belong to the same group.
            if prob > args.grm_p:
                grm_compressions_list.append(node)
                for optional_node in compressions_list:
                    if optional_node['group'] == node['group']:
                        grm_compressions_list.append(optional_node)

    return grm_compressions_list


def run_compression(args, model, tokenizer, parse_list, ext_sents, compressions_list):
    """
    Runs compression model. From the extracted sentences, uses a constituency tree
    to enumerate compression options, which the model then chooses from to delete.
    The resulting summary is optionally post-processed with linguistic rules.
    """

    # Split compressions list into optional and non-optional compressions. Only the
    # non-optional compressions will be used for evaluation. Among the options
    # that are selected, the optional compressions that belong to the same group
    # will also be selected post-hoc.
    eval_compressions_list = [
        node for node in compressions_list if not node['optional']
    ]

    try:
        # Encode inputs, create batch, and run compression model.
        (
            input_ids,
            segment_ids,
            attention_mask,
            span_ids,
            label_ids,
        ) = encode_compression_inputs(
            ext_sents,
            eval_compressions_list,
            tokenizer,
            args.max_seq_length,
            args.max_span_length,
            args.device,
            do_pad=True,
        )

        batch = {
            'input_ids': input_ids.unsqueeze(0),
            'segment_ids': segment_ids.unsqueeze(0),
            'attention_masks': attention_mask.unsqueeze(0),
            'span_ids': span_ids.unsqueeze(0),
            'label_ids': label_ids.unsqueeze(0),
        }

        with torch.no_grad():
            logits = model(batch)

        eval_compressions_list = eval_compressions_list[:args.max_span_length]
        # Obtain deletion probabilities for each span.
        probs = F.softmax(logits, 1)[0, 1, :]
        span_probs = unpack(probs.masked_select(label_ids != -1))
    except:
        span_probs = []

    node_list = []
    label_list = [[0 for _ in range(len(sent))] for sent in ext_sents]

    if len(span_probs) > 0:
        assert len(eval_compressions_list) == len(span_probs)

        # Set word-level labels for spans marked for deletion.
        for (node, prob) in zip(eval_compressions_list, span_probs):
            if prob > args.p:
                node['score'] = prob
                node_list.append(node)
                apply_compression(label_list, node)

        # For all selected nodes, find optional nodes that belong to the
        # same group, if any. Delete these nodes as well.
        orig_node_list_len = len(node_list)
        for node in node_list[:orig_node_list_len]:
            optional_node_list = [
                optional_node for optional_node in compressions_list
                if (
                    optional_node['optional']
                    and optional_node['group'] == node['group']
                )
            ]
            for optional_node in optional_node_list:
                node['score'] = -1
                node_list.append(optional_node)
                apply_compression(label_list, optional_node)

    # Post-process compressed sentences for grammaticality.
    # ext_sents, label_list = postprocess_doc(ext_sents, label_list)
    _check_dims(ext_sents, label_list)

    # Build up doc with words that are kept.
    cmp_sents = []
    for (sent_list, sent_labels) in zip(ext_sents, label_list):
        token_list = [
            word for (word, label)
            in zip(sent_list, sent_labels) if label == 0
        ]
        if token_list:
            cmp_sents.append(token_list)

    # Only keep [sent_index, node_index] in node_list.
    # node_list = [
    #     [node['sent_index'], node['node_index']] for node in node_list
    # ]

    return (cmp_sents, label_list, node_list)


def color_compressions(ext_sents, label_list):
    """Colors deleted spans in extracted sentences with compression labels."""

    colored_tokens = []

    i = 0
    while i < len(label_list):
        j = 0
        while j < len(label_list[i]):
            if label_list[i][j] == 1:
                span = []
                k = j
                while k < len(label_list[i]) and label_list[i][k] == 1:
                    span.append(ext_sents[i][k])
                    k += 1
                colored_tokens.append(
                    colored(' '.join(span), 'red', attrs=['bold', 'underline'])
                )
                j = k
            else:
                colored_tokens.append(ext_sents[i][j])
                j += 1
        i += 1

    return ' '.join(colored_tokens)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--ext_model', type=str, default='google/electra-base-discriminator')
    parser.add_argument('--cmp_model', type=str, default='google/electra-base-discriminator')
    parser.add_argument('--ext_ckpt_path', type=str)
    parser.add_argument('--cmp_ckpt_path', type=str)
    parser.add_argument('--grm_ckpt_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--p', type=float, default=0.3)
    parser.add_argument('--grm_p', type=float, default=0.5)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--max_label_length', type=int, default=50)
    parser.add_argument('--max_span_length', type=int, default=100)
    parser.add_argument('--trigram_block', action='store_true', default=False)
    parser.add_argument('--do_compress', action='store_true', default=False)
    parser.add_argument('--do_grammar', action='store_true', default=False)
    parser.add_argument('--do_viz', action='store_true', default=False)
    parser.add_argument('--quick_test', action='store_true', default=False)

    args = parser.parse_args()
    print(args)
    print()

    ext_tokenizer = AutoTokenizer.from_pretrained(args.ext_model)
    ext_model = cuda(ExtractionModel(args.ext_model))
    ext_model.load_state_dict(torch.load(args.ext_ckpt_path, map_location=f'cuda:{args.device}'))
    ext_model.eval()
    print('loaded extraction model')

    if args.do_compress:
        cmp_tokenizer = AutoTokenizer.from_pretrained(args.cmp_model)
        cmp_model = cuda(CompressionModel(args.cmp_model, 0.))
        cmp_model.load_state_dict(torch.load(args.cmp_ckpt_path, map_location=f'cuda:{args.device}'))
        cmp_model.eval()
        print('loaded compression model')

    if args.do_grammar:
        grm_tokenizer = AutoTokenizer.from_pretrained(args.cmp_model)
        grm_model = cuda(CompressionModel(args.cmp_model, 0.))
        grm_model.load_state_dict(torch.load(args.grm_ckpt_path, map_location=f'cuda:{args.device}'))
        grm_model.eval()
        print('loaded grammar model')

    test_elems = load_elems(args.test_path)
    if args.quick_test:
        random.shuffle(test_elems)
        test_elems = test_elems[:100]
    
    test_loader = test_elems
    if not args.do_viz:
        test_loader = tqdm(test_elems, ncols=100)

    hyp_list = []
    ref_list = []
    rate_list = []

    if args.output_path:
        output_elems = []

    for i, elem in enumerate(test_loader):
        abs_list = elem['abs_list']
        doc_list = elem['doc_list']
        parse_list = elem['parse_list']

        # Filter inputs.
        assert len(doc_list) == len(parse_list)
        _doc_list = []
        _parse_list = []
        for (sent_list, parse) in zip(doc_list, parse_list):
            if len(sent_list) > 5 and parse != '':
                _doc_list.append(sent_list[:200])
                _parse_list.append(parse)
        doc_list = _doc_list
        parse_list = _parse_list

        # Run extraction model.
        ext_sents, ext_sents_indices = run_extraction(
            args=args,
            model=ext_model,
            tokenizer=ext_tokenizer,
            doc_list=doc_list,
        )

        # For lead/oracle computation.
        # ext_sents = doc_list
        # ext_sents_indices = list(range(len(ext_sents)))

        _check_dims(ext_sents, ext_sents_indices)

        # Using extracted sentences and abstractive summary,
        # build up strings for evaluation.
        hyp = ext_sents
        ref = abs_list

        compressions_list = []

        if args.do_compress:
            cmp_sents = []
            label_list = []

            # Run compression model on each sentence individually.
            for (ext_sent, ext_sent_index) in zip(ext_sents, ext_sents_indices):
                _compressions_list = derive_compressions(
                    parse_list, [ext_sent], [ext_sent_index]
                )

                if args.do_grammar:
                    _compressions_list = run_grammar(
                        args=args,
                        model=grm_model,
                        tokenizer=grm_tokenizer,
                        parse_list=parse_list,
                        ext_sents=[ext_sent],
                        compressions_list=_compressions_list,
                    )

                _cmp_sents, _label_list, _compressions_list = run_compression(
                    args=args,
                    model=cmp_model,
                    tokenizer=cmp_tokenizer,
                    parse_list=parse_list,
                    ext_sents=[ext_sent],
                    compressions_list=_compressions_list,
                )
                compressions_list.append(_compressions_list)

                cmp_sents.extend(_cmp_sents)
                label_list.extend(_label_list)

            _check_dims(ext_sents, label_list)

            # Using extracted sentences with span-based compression and
            # abstractive summary, build up strings for evaluation.
            hyp = cmp_sents
            ref = abs_list
            rate_list.append(compute_rate(label_list))

        if args.do_viz:
            print(color_compressions(ext_sents, label_list))
            print()

        hyp_str = [' '.join(sent) for sent in hyp]
        ref_str = [' '.join(sent) for sent in ref]

        if args.output_path:
            output_elem = {
                'name': elem['name'],
                'part': elem['part'],
                'compressions': compressions_list,
                'orig_hyp_list': [' '.join(sent) for sent in ext_sents],
                'hyp_list': hyp_str,
                'ref_list': ref_str,
            }
            output_elems.append(output_elem)

            if i == 0:
                print(output_elem)

            
        hyp_list.append(hyp_str)
        ref_list.append([ref_str])

    if args.output_path:
        write_elems(output_elems, args.output_path)
        print(f'outputs written to \'{args.output_path}\'')

    # Compute ROUGE metrics.
    rouge1, rouge2, rougel = compute_perl_rouge(hyp_list, ref_list)
    rate = (
        round(np.mean(rate_list) * 100., 1)
        if len(rate_list) else 0.
    )

    # Build results.
    results_dict = {
        'CR (%)': rate,
        'ROUGE-1': rouge1,
        'ROUGE-2': rouge2,
        'ROUGE-L': rougel,
    }

    print_results(results_dict)
