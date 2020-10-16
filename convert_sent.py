import argparse
import os

from utils import load_elems, write_elems
from tqdm import tqdm


def main(args):
    dataset = args.dataset

    for split in ('train', 'dev', 'test'):
        dir_path = f'summ_data/{dataset}_sent'
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        elems = load_elems(f'summ_data/{dataset}/{split}.txt')
        ext_elems = load_elems(f'summ_data/{dataset}/{split}_ext.txt')

        ext_elem_ids = set([ext_elem['name'] for ext_elem in ext_elems])
        elems = [elem for elem in elems if elem['name'] in ext_elem_ids]
        assert len(elems) == len(ext_elems)

        out_elems = []
        out_ext_elems = []

        for (elem, ext_elem) in tqdm(zip(elems, ext_elems)):
            assert elem['name'] == ext_elem['name']

            name = elem['name']
            part = f'{dataset}_sent'
            abs_list = elem['abs_list']

            label_list = ext_elem['label_list']
            _doc_list = elem['doc_list']
            _parse_list = elem['parse_list']

            if len(_parse_list) == 0:
                _parse_list = ['']

            doc_list = []
            parse_list = []
            
            # Take sentences according to extractive oracle.
            for (label, doc, parse) in zip(label_list, _doc_list, _parse_list):
                if label == 1:
                    doc_list.append(doc)
                    parse_list.append(parse)

            # Create per-sentence dataset where the extractive oracle is hardcoded to
            # always pick up that sentence.
            for (sent, parse) in zip(doc_list, parse_list):
                out_elems.append({'name': name, 'part': part, 'abs_list': abs_list, 'doc_list': [sent], 'parse_list': [parse]})
                out_ext_elems.append({'name': name, 'part': part, 'label_list': [1]})

        write_elems(out_elems, f'summ_data/{dataset}_sent/{split}.txt')
        write_elems(out_ext_elems, f'summ_data/{dataset}_sent/{split}_ext.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    main(args)
