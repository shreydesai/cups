import argparse
import random
import uuid

from tqdm import tqdm

from compression_utils import build_tree, find_compressions, rules
from utils import load_elems, write_elems


def compression_rate(sent_labels):
    return sum(sent_labels) / len(sent_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    print(args)
    print()

    random.seed(0)

    doc_elems = load_elems(args.input_path)
    sent_elems = []

    total = 0

    for doc_elem in tqdm(doc_elems):
        name = doc_elem['name']
        part = doc_elem['part']

        doc_list = doc_elem['doc_list']
        tree_list = doc_elem['doc_parse']
        label_list = doc_elem['label_list']

        # Find candidate sentences that meet compression threshold
        candidate_sents = [
            (sent_list, sent_parse, sent_labels)
            for (sent_list, sent_parse, sent_labels) in zip(
                doc_list, tree_list, label_list
            )
            if (
                len(sent_list) < 128
                and 0.2 < compression_rate(sent_labels) < 0.8
            )
        ]

        for (sent_list, sent_parse, sent_labels) in candidate_sents:
            sent_elem = {
                'name': str(uuid.uuid4().hex),
                'doc_name': name,
                'part': 'debate_sent',
                'abs_list': [
                    [
                        token
                        for (token, label) in zip(sent_list, sent_labels)
                        if label == 1
                    ],
                ],
                'doc_list': [sent_list],
                'doc_parse': [sent_parse],
                'label_list': [sent_labels],
            }
            sent_elems.append(sent_elem)

    random.shuffle(sent_elems)
    
    write_elems(sent_elems, args.output_path) 
    print(f'wrote {len(sent_elems)} elems to \'{args.output_path}\'')
