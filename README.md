# Compressive Summarization with Plausibility and Salience Modeling
 
Code and datasets for our EMNLP 2020 paper [Compressive Summarization with Plausibility and Salience Modeling](https://arxiv.org/pdf/2010.07886.pdf). If you found this project helpful, please consider citing our paper:
 
```bibtex
@inproceedings{desai-etal-2020-compressive,
  author={Desai, Shrey and Xu, Jiacheng and Durrett, Greg},
  title={{Compressive Summarization with Plausibility and Salience Modeling}},
  year={2020},
  booktitle={Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)},
}
```
 
## Abstract
 
Compressive summarization systems typically rely on a crafted set of syntactic rules to determine what spans of possible summary sentences can be deleted, then learn a model of what to actually delete by optimizing for content selection (ROUGE). In this work, we propose to relax the rigid syntactic constraints on candidate spans and instead leave compression decisions to two data-driven criteria: plausibility and salience. Deleting a span is plausible if removing it maintains the grammaticality and factuality of a sentence, and spans are salient if they contain important information from the summary. Each of these is judged by a pre-trained Transformer model, and only deletions that are both plausible and not salient can be applied. When integrated into a simple extraction-compression pipeline, our method achieves strong in-domain results on benchmark summarization datasets, and human evaluation shows that the plausibility model generally selects for grammatical and factual deletions. Furthermore, the flexibility of our approach allows it to generalize cross-domain: our system fine-tuned on only 500 samples from a new domain can match or exceed an in-domain extractive model trained on much more data.
 
## Project Setup
 
This project requires Python 3.6+ and has the following requirements:
- numpy==1.18.1
- py-rouge==1.1
- pythonrouge==0.2
- scikit-learn==0.22.1
- termcolor==1.1.0
- transformers==2.8.0
- torch==1.2.0
- tqdm==4.42.1
 
Additionally, we provide a link to our (tokenized) [summarization datasets](https://drive.google.com/file/d/1btXVd7kqfHjzTyk_51XQj1uHwSy8x-X3/view?usp=sharing); it can be unzipped with `tar -zxf summ_data.tar.gz`. Inside, there are document-level datasets (cnndm, cnn, curation, google, multinews, nyt, pubmed, reddit, wikihow, xsum) and sentence-level datasets (*_sent). The former is used for extraction training while the latter is used for compression training.
 
Each dataset contains a train, dev, and test split. A single sample is represented as JSON object with the following keys:
- `name`: unique sample id
- `abs_list`: sentence-separated list of tokens in the summary
- `doc_list`: sentence-separated list of tokens in the document
- `parse_list`: list of constituency trees (one for each sentence in the document)
 
## Training and Evaluating Models
 
Our models are trained and evaluated on an NVIDIA 32GB V100 GPU. If your GPU has less memory, consider reducing the `batch_size` and increasing `grad_accumulation_steps`. Additionally, we open-source our [model checkpoints](https://drive.google.com/file/d/1E-gti3IOJCsXOG4GlOcFzeXg4Dq6xQbY/view?usp=sharing).
### Extraction Training
 
The following command fine-tunes an ELECTRA-Base sentence extractive model on CNN:
 
```bash
$ export DATASET="cnn"
$ python3 train_extraction.py \
	--device 0 \
	--model "google/electra-base-discriminator" \
	--ckpt_path "ckpt/${DATASET}/electra_ext.pt" \
	--train_path "summ_data/${DATASET}/train.txt" \
	--dev_path "summ_data/${DATASET}/dev.txt" \
	--max_train_steps 50000 \
	--batch_size 16 \
	--learning_rate 1e-5 \
	--eval_interval 1000 \
	--max_grad_norm 1.0 \
	--do_train
```
 
### Compression Training
 
The compression model architecture is identical between the plausibility and salience models; the only difference is the data they are trained on. Plausibility leverages the Google dataset, while salience leverages any non-Google dataset (e.g., CNN). The following command fine-tunes an ELECTRA-Base span-based compression model on CNN:
 
```bash
$ export DATASET="cnn"
$ python3 train_compression.py \
	--device 0 \
	--model "google/electra-base-discriminator" \
	--ckpt_path "ckpt/${DATASET}/electra_cmp.pt" \
	--train_path "summ_data/${DATASET}_sent/train.txt" \
	--dev_path "summ_data/${DATASET}_sent/dev.txt" \
	--max_train_steps 50000 \
	--batch_size 16 \
	--learning_rate 1e-5 \
	--eval_interval 1000 \
	--max_grad_norm 1.0 \
	--do_train
```
 
### Pipelined Evaluation
 
Once we fine-tune 3 ELECTRA-Base models (extraction, plausibility, and salience), we can put them together in a pipeline for evaluation on a summarization test set.
 
Note that the default hyperparameters are already included in the `eval_pipeline.py` arguments, but we found 0.6 to work well for plausibility (`--grm_p`) and the following values for salience (`--p`):
| CNNDM | CNN | WikiHow | XSum | Reddit |
|--|--|--|--|--|
| 0.7 | 0.5 | 0.45 | 0.6 | 0.7 |
 
The following command evaluates our compressive system on CNN:
 
```bash
$ export DATASET="cnn"
$ python3 eval_pipeline.py \
	--device 0 \
	--ext_model "google/electra-base-discriminator" \
	--cmp_model "google/electra-base-discriminator" \
	--ext_ckpt_path "ckpt/${DATASET}/electra_ext.pt" \
	--cmp_ckpt_path "ckpt/${DATASET}/electra_cmp.pt" \
	--grm_ckpt_path "ckpt/google/electra_cmp.pt" \
	--test_path "summ_data/${DATASET}/test.txt" \
	--k 3 \
	--do_compress \
	--do_grammar
```
 
You may tune `grm_p` for grammaticality: the higher this value is, the more restrictive the plausibility model gets when deleting spans, but as a result, the overall compressive ROUGE will converge towards the extractive ROUGE. This may not necessarily be a bad thing if you care a lot about grammaticality and factuality.
