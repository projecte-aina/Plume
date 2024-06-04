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

### Inference

The following script is used to translate the Flores-200 dataset using the trained model. Some variables must be defined. Specifically: `checkpoint, name, vocab_size, model_dir`. For inference we use beam search with a beam size of 5 limiting the number of tokens to 512.

```bash
bash ./inference/experiments.sh
```

Translations will be saved in `./inference/translations/` folder.

For running experiments as outlined in section 4.2 of the paper, which evaluates models without indicating the source tag, utilize the provided script:

```bash
bash ./inference/experiments_ignore_src.sh
```

# Attention analysis

For computing the coverage we provide the following script which computes coverage for each head using Flores-200 dataset. Some variables must be defined: `checkpoint, name, model_dir`.

```bash
bash ./attention_analysis/run_experiments.sh
```

This code will save as .npy files the coverage matrices which are then visualized in the following jupyter notebook: `./attention_analysis/Att_metrics_Plots.ipynb`.

We also provide the following script to plot the attention matrices of the first sentence from Flores-200 dataset:

```bash
bash ./attention_analysis/get_att_matrix.sh
```

Resulting plots will be saved in the corresponding folder inside `./results`.

### Heads masking

We provide the code to compute the heads masking experiments from section 4.2 in the paper. Note that to mask heads we must first compute the coverage metrics as detailed in previous section and the variable `ATT_ANALYSIS_FULL_PATH` must be defined accordingly (`./attention_analysis/results`).

```bash
bash ./heads_masking/experiments.sh
```

This code will save the corresponding plots and translation examples in `./heads_masking/results` folder. We provide a jupyter notebook to visualize the results: `./heads_masking/Heads_Masking_Plots.ipynb`.

### Representation space


## Citation

```bibtex

```


## License
