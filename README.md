# Parlam

This repository contains the code for the paper "Investigating the translation capabilities of Large Language Models trained
on parallel data only". The preprint is available on [arXiv]().

## About Parlam

In recent years, Large Language Models (LLMs) have demonstrated exceptional proficiency across a broad spectrum of Natural Language Processing (NLP) tasks, including Machine Translation. However, previous methodologies predominantly relied on iterative processes such as instruction fine-tuning or continual pre-training, leaving unexplored the challenges of training LLMs solely on parallel data. In this work, we introduce Parlam (PARallel LAnguage Model), a collection of three 2B LLMs featuring varying vocabulary sizes (32k, 128k, and 256k) trained exclusively on  Catalan-centric parallel examples. These models perform comparable to previous encoder-decoder architectures on 16 supervised translation directions and 56 zero-shot ones. Utilizing this set of models, we conduct a thorough investigation into the translation capabilities of LLMs, probing their performance, the impact of the different elements of the prompt, and their cross-lingual representation space.

## Models Description

## Running Parlam

```python
from transformers import 

tokenizer = 
model =

```

## Running experiments

Install dependencies:

```bash
pip install -r requirements.txt
```

### Tokenizer

The following scripts will create the tokenizers. Note that the folder `./tokenizer/samplings/` must contain a txt file for each language.

```bash
bash ./tokenizer/create_tokenizer_over_eus_deu_eng_1M.sh
bash ./tokenizer/create_tokenizer_equal_1M.sh
```

The following script will compute all the metrics used to evaluate the tokenizers and will save it in `./tokenizer/assets/`

```bash
bash ./tokenizer/compute_tokenizations.sh
```

Results are visualized in the following jupyter notebook: `./tokenizer/Fertility_Plots.ipynb`


### Training

The following scripts will execute model training using DeepSpeed (ZeRO stage 2). For training we used 40 NVIDIA H100-64GB GPUs with full float32 precision. Note that some variables must be defined in the script, namely: `HF_DATASETS_CACHE, HF_HOME, TOKENIZER_PATH, VOCAB_SIZE, DATASET_PATH`. This code will automatically tokenize the data given a HF dataset.

```bash
bash ./training/parlam_distributed.sh
```

DeepSpeed checkpoints will be saved in `./training/output/` folder that can be converted to HF checkpoints using the following script:

```bash
bash ./training/output/convert.sh
```

Converting DeepSpeed checkpoints is required to run the remaining experiments.

## Citation

```bibtex

```


## License
