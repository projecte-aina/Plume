<div align="center">

## Plume: the first LLM trained for NMT with only parallel Catalan-Centric data from scratch

#### [Javier García](https://www.bsc.es/ca/garcia-gilabert-javier), [Carlos Escolano](https://www.bsc.es/es/escolano-peinado-carlos), [Aleix Sant](https://www.bsc.es/es/sant-savall-aleix), [Francesca De Luca](https://www.bsc.es/de-luca-fornaciari-francesca), [Audrey Mash](https://www.bsc.es/mash-audrey) and [Maite Melero](https://www.bsc.es/melero-nogues-maite)

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![license](https://img.shields.io/badge/License-Apache_2.0-blue.svg)]()

</div>

> This repository contains the code for the paper "Investigating the translation capabilities of Large Language Models trained
on parallel data only". The preprint is available on [arXiv]().

#### Abstract

In recent years, Large Language Models (LLMs) have demonstrated exceptional proficiency across a broad spectrum of Natural Language Processing (NLP) tasks, including Machine Translation. However, previous methodologies predominantly relied on iterative processes such as instruction fine-tuning or continual pre-training, leaving unexplored the challenges of training LLMs solely on parallel data. In this work, we introduce Parlam (PARallel LAnguage Model), a collection of three 2B LLMs featuring varying vocabulary sizes (32k, 128k, and 256k) trained exclusively on  Catalan-centric parallel examples. These models perform comparable to previous encoder-decoder architectures on 16 supervised translation directions and 56 zero-shot ones. Utilizing this set of models, we conduct a thorough investigation into the translation capabilities of LLMs, probing their performance, the impact of the different elements of the prompt, and their cross-lingual representation space.

## Models Description

Plume is the first LLM trained for Neural Machine Translation with only parallel Catalan-Centric data from scratch. It is a language model with the same architecture as Gemma 2B. The model is trained for general translation tasks at sentence level. Information about training, architecture and interpretability of the model are described in the paper.

- **Developed by:** Machine Translation Unit at the Barcelona Supercomputing Center (BSC).
- **Languages:** Spanish, French, Italian, Portuguese, Galician, German, English, and Basque.
- **License:** Apache License, Version 2.0

## Running the models

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

The following script will execute model training using DeepSpeed (ZeRO stage 2). For training we used 40 NVIDIA H100-64GB GPUs with full float32 precision. Note that some variables must be defined in the script, namely: `HF_DATASETS_CACHE, HF_HOME, TOKENIZER_PATH, VOCAB_SIZE, DATASET_PATH`. This code will automatically tokenize the data given a HF dataset.

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

### Attention analysis

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

#### Distances 

We provide the scripts to compute the distances between layers. First, we extract the model's representations using the following script. Note that some variables must be pre-defined: `model_dir, name_model`.

```bash
bash ./representation_space/extract_representations.sh
```

This will save the extracted token representations for each language as numpy files in `./representation_space/results` folder. Then, to compute distances we provide the following script:

```bash
bash ./representation_space/compute_distances.sh
```

Pairwise distances will be saved in the corresponding folder inside `./representation_space/results` folder. We provide a jupyter notebook to visualize the computed distances: `./representation_space/Distances_Plots.ipynb`.

#### Visualization

To visualize token embeddings as done in section 4.3 in the paper we provide the following script which computes UMAP 2D and Spherical Voronoi Diagrams for the token representations:

```bash
bash ./representation_space/compute_umap.sh
```

Pairwise distances will be saved in the corresponding folder inside `./representation_space/results` folder. We provide a jupyter notebook to visualize the computed UMAP latent variables: `./representation_space/UMAP_Plots.ipynb` and another jupyter notebook to create the Spherical Voronoi Diagrams: `./representation_space/Voronoi_Plots.ipynb`.

In addition, we provide as a zip file with some Spherical Voronoi Diagrams that have already computed for the 32k model variant: `./representation_space/voronoi_plots_32k.zip`.

### Vocabulary overlap

We provide the script to compute the vocabulary overlap between pair of languages:

```bash
bash ./voc_overlap/compute_overlapping.sh
```

### Evaluation

TO-DO

## Citation

```bibtex

```

## Acknowledgements


## Contact

Please feel free to write us at with any questions you may have to {javier.garcia1, carlos.escolano, aleix.santsavall, francesca.delucafornaciari, audrey.mash, maite.melero}@bsc.es 

## License
