import os
import json
import re

import torch

from pythonrouge.pythonrouge import Pythonrouge
from rouge import Rouge
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score


perl_remap = {
    '-lrb-': '(',
    '-rrb-': ')',
    '-lcb-': '{',
    '-rcb-': '}',
    '-lsb-': '[',
    '-rsb-': ']',
    '``': '"',
    "''": '"',
}


def print_results(results_dict):
    for k, v in results_dict.items():
        print(f'{k} = {v}')


def get_oracle_path(path, oracle_type, suffix=''):
    dir_path, file_name = os.path.split(path)
    file_name = file_name.split('.')[0]
    oracle_file_path = (
        f'{file_name}_{oracle_type}_{suffix}.txt'
        if suffix else f'{file_name}_{oracle_type}.txt'
    )
    oracle_path = os.path.join(dir_path, oracle_file_path)
    if not os.path.exists(oracle_path):
        raise RuntimeError(f'Path \'{oracle_path}\' does not exist!')
    return oracle_path


def load_elems(path):
    with open(path) as f:
        return [
            json.loads(l.rstrip())
            for l in tqdm(f, desc=f'loading \'{path}\'')
        ]


def write_elems(elems, path):
    with open(path, 'w+') as f:
        for elem in elems:
            try:
                json.dumps(elem)
            except:
                print(elem)
                import sys;sys.exit(1)
            f.write(f'{json.dumps(elem)}\n')


def ngrams(tokens, n=1):
    return set(zip(*(tokens[i:] for i in range(n))))


def flatten(x):
    return [z for y in x for z in y]


def limit_doc(doc_list, limit=512):
    index = 0
    n_tokens = 0
    for sent in doc_list:
        index += 1
        n_tokens += len(sent)
        if n_tokens > limit:
            break
    return doc_list[:index]


def build_string(elem):
    if len(elem) == 0:
        return ''
    if isinstance(elem[0], list):
        elem = [token for token in flatten(elem) if token]
    return ' '.join(elem)


def postprocess_doc(doc_list, doc_labels):
    """Uses rules to ensure document fluency after syntactic compression."""

    for i in range(len(doc_list)):
        sent_list = doc_list[i]
        sent_labels = doc_labels[i]
        for j in range(len(doc_list[i])):
            # Remove commas in beginning
            if all(sent_labels[:j]) and sent_list[j] == ',':
                doc_labels[i][j] = 1
            
            # Remove commas in ending
            if (
                all(sent_labels[(j + 1):-1])
                and sent_list[j] == ','
                and sent_list[-1] == '.'
            ):
                doc_labels[i][j] = 1

            # Remove joint commas
            if j < len(sent_list) - 1 and sent_list[j] in (',', '.',):
                # Skip span of deletions and evaluate next word
                k = j + 1
                while k < len(sent_list) and sent_labels[k] == 1:
                    k += 1
                if (
                    k < len(sent_list)
                    and (
                        (sent_list[j] == ',' and sent_list[k] == ',')
                        or (sent_list[j] == '.' and sent_list[k] == '.')
                    )
                ):
                    doc_labels[i][j] = 1
                    doc_labels[i][k] = 1

            # Match a/an 
            if j < len(sent_list) - 1 and sent_list[j] in ('a', 'an'):
                # Skip span of deletions and evaluate next word
                k = j + 1
                while k < len(sent_list) and sent_labels[k] == 1:
                    k += 1
                if k < len(sent_list):
                    if sent_list[j] == 'a' and sent_list[k][0] in (
                        'a', 'e', 'i', 'o', 'u', 'h'
                    ):
                        doc_list[i][j] = 'an'
                    elif sent_list[j] == 'an' and sent_list[k][0] not in (
                        'a', 'e', 'i', 'o', 'u', 'h'
                    ):
                        doc_list[i][j] = 'a'

    return (doc_list, doc_labels)


def compute_rouge(hyp_ngrams, ref_ngrams):
    assert isinstance(hyp_ngrams, set) and isinstance(ref_ngrams, set)
    overlap = len(hyp_ngrams & ref_ngrams)
    p = overlap / len(hyp_ngrams) if len(hyp_ngrams) > 0 else 0.
    r = overlap / len(ref_ngrams) if len(ref_ngrams) > 0 else 0.
    return 2. * ((p * r) / (p + r + 1e-8))


def compute_token_metrics(true_list, pred_list, t_list=[0.5]):
    auroc = round(roc_auc_score(true_list, pred_list) * 100., 2)
    aupr = round(average_precision_score(true_list, pred_list) * 100., 2)
    f1_list = []
    for t in t_list:
        _pred_list = [1 if p > t else 0 for p in pred_list]
        f1 = round(f1_score(true_list, _pred_list) * 100., 2)
        f1_list.append(f1)
    if len(f1_list) == 1:
        return (auroc, aupr, f1_list[0])
    return (auroc, aupr, f1_list)


def compute_approx_rouge(hyp_list, ref_list):
    rouge = Rouge(
        metrics=['rouge-n'],
        max_n=2,
        limit_length=False,
        apply_avg=True,
        alpha=0.5,
        stemming=True,
    )

    scores = rouge.get_scores(hyp_list, ref_list)
    rouge_1_f = round(scores['rouge-1']['f'] * 100., 2)
    rouge_2_f = round(scores['rouge-2']['f'] * 100., 2)

    return (rouge_1_f, rouge_2_f)


def compute_perl_rouge(hyp_list, ref_list):
    def clean(x):
        return re.sub(
            r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
            lambda m: perl_remap.get(m.group()), x
        )

    def preprocess(doc_list, reference=False):
        doc_list_pre = []
        if reference:
            doc_list = doc_list[0]
        for sent_str in doc_list:
            sent_str = clean(sent_str.lower())
            doc_list_pre.append(sent_str)
        if reference:
            doc_list_pre = [doc_list_pre]
        return doc_list_pre


    rouge = Pythonrouge(
        summary_file_exist=False,
        summary=[preprocess(hyp) for hyp in hyp_list],
        reference=[preprocess(ref, reference=True) for ref in ref_list],
        n_gram=2,
        ROUGE_SU4=False,
        ROUGE_L=True,
        ROUGE_W=False,
        ROUGE_W_Weight=1.2,
        recall_only=False,
        f_measure_only=False,
        stemming=True,
        stopwords=False,
        word_level=False,
        length_limit=False,
        length=50,
        use_cf=True,
        cf=95,
        scoring_formula='average',
        resampling=True,
        samples=1000,
        favor=False,
        p=0.5,
    )

    scores = rouge.calc_score()
    rouge_1_f = round(scores['ROUGE-1-F'] * 100., 2)
    rouge_2_f = round(scores['ROUGE-2-F'] * 100., 2)
    rouge_l_f = round(scores['ROUGE-L-F'] * 100., 2)
    return (rouge_1_f, rouge_2_f, rouge_l_f)


### span extractor


def get_device_of(tensor: torch.Tensor):
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


def get_range_vector(size, device):
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)


def batched_span_select(target, spans):
    span_starts, span_ends = spans.split(1, dim=-1)
    span_widths = span_ends - span_starts
    max_batch_span_width = span_widths.max().item() + 1
    max_span_range_indices = get_range_vector(max_batch_span_width, get_device_of(target)).view(
        1, 1, -1
    )
    span_mask = (max_span_range_indices <= span_widths).float()
    raw_span_indices = span_ends - max_span_range_indices
    span_mask = span_mask * (raw_span_indices >= 0).float()
    span_indices = torch.nn.functional.relu(raw_span_indices.float()).long()
    span_embeddings = batched_index_select(target, span_indices)
    return span_embeddings, span_mask


def flatten_and_batch_shift_indices(indices, sequence_length):
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise ConfigurationError(
            f"All elements in indices should be in range (0, {sequence_length - 1})"
        )
    offsets = get_range_vector(indices.size(0), get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)
    offset_indices = indices + offsets
    offset_indices = offset_indices.view(-1)
    return offset_indices


def batched_index_select(target, indices, flattened_indices=None):
    if flattened_indices is None:
        flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))
    flattened_target = target.view(-1, target.size(-1))
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets


def weighted_sum(matrix, attention):
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)


def masked_softmax(vector, mask, dim=-1, memory_efficient=False, mask_fill_value=-1e32):
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).to(dtype=torch.bool), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


class TimeDistributed(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self._module = module

    def forward(self, *inputs, pass_through=None, **kwargs):
        pass_through = pass_through or []
        reshaped_inputs = [self._reshape_tensor(input_tensor) for input_tensor in inputs]
        some_input = None
        if inputs:
            some_input = inputs[-1]
        reshaped_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and key not in pass_through:
                if some_input is None:
                    some_input = value
                value = self._reshape_tensor(value)
            reshaped_kwargs[key] = value
        reshaped_outputs = self._module(*reshaped_inputs, **reshaped_kwargs)
        if some_input is None:
            raise RuntimeError("No input tensor to time-distribute")
        new_size = some_input.size()[:2] + reshaped_outputs.size()[1:]
        outputs = reshaped_outputs.contiguous().view(new_size)
        return outputs

    @staticmethod
    def _reshape_tensor(input_tensor):
        input_size = input_tensor.size()
        if len(input_size) <= 2:
            raise RuntimeError(f"No dimension to distribute: {input_size}")
        squashed_shape = [-1] + list(input_size[2:])
        return input_tensor.contiguous().view(*squashed_shape)


class SelfAttentiveSpanExtractor(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self._input_dim = input_dim
        self._global_attention = TimeDistributed(torch.nn.Linear(input_dim, 1))

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._input_dim

    def forward(self, sequence_tensor, span_indices, span_indices_mask=None):
        global_attention_logits = self._global_attention(sequence_tensor)
        concat_tensor = torch.cat([sequence_tensor, global_attention_logits], -1)
        concat_output, span_mask = batched_span_select(concat_tensor, span_indices)
        span_embeddings = concat_output[:, :, :, :-1]
        span_attention_logits = concat_output[:, :, :, -1]
        span_attention_weights = masked_softmax(span_attention_logits, span_mask)
        attended_text_embeddings = weighted_sum(span_embeddings, span_attention_weights)
        if span_indices_mask is not None:
            return attended_text_embeddings * span_indices_mask.unsqueeze(-1).float()
        return attended_text_embeddings
