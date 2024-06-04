#!/usr/bin/env python
# Implementation adapted from Petrov, A., La Malfa, E., Torr, P., & Bibi, A. (2024). 
# Language model tokenizers introduce unfairness between languages. Advances in 
# Neural Information Processing Systems, 36.
# https://github.com/AleksandarPetrov/tokenization-fairness

import multiprocessing
from typing import Type
import pandas
import os
from collections import defaultdict
import tqdm
from tokenizers import Tokenizer
from tokenizer_interface import HuggingFaceTokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

tokenizers_names = [  ('unigram', 'sampling_equal', 1, 32000), ('unigram', 'sampling_equal', 1, 128000), ('unigram', 'sampling_equal', 1, 256000),
                      ('bpe', 'sampling_equal', 1, 32000), ('bpe', 'sampling_equal', 1, 128000), ('bpe', 'sampling_equal', 1, 256000),
                      ('unigram', 'sampling_over_eus_deu_eng', 1, 32000), ('unigram', 'sampling_over_eus_deu_eng', 1, 128000), ('unigram', 'sampling_over_eus_deu_eng', 1, 256000),
                      ('bpe', 'sampling_over_eus_deu_eng', 1, 32000), ('bpe', 'sampling_over_eus_deu_eng', 1, 128000), ('bpe', 'sampling_over_eus_deu_eng', 1, 256000),
                    ]

tokenizers_paths = ['./tokenizers/{}.{}_{}M/size_{}/tokenizer_fast/'.format(type_alg, sampling, size, vocab_size) for type_alg, sampling, size, vocab_size in tokenizers_names]

ALL_TOKENIZERS = []
for path, name in zip(tokenizers_paths, tokenizers_names):
    type_alg, sampling, size, vocab_size = name[0], name[1], name[2], name[3]
    tokenizer_hf = HuggingFaceTokenizer(path, type_alg, sampling, size, vocab_size)
    ALL_TOKENIZERS.append(tokenizer_hf)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

DATASET_PATH = "./flores200_dataset"

langs = ['spa_Latn', 'cat_Latn','por_Latn', 'ita_Latn', 'glg_Latn', 'fra_Latn', 'eus_Latn', 'eng_Latn', 'deu_Latn']
print(f"Using {len(langs)} languages in the dataset:")
for lang in langs:
    print(f"\t{lang}")

tokens_per_sentence = {tk.pretty_name: {lang:[] for lang in langs} for tk in ALL_TOKENIZERS}
# get the RTL languages (lines in flores_rtl.txt    ):
with open("./flores_rtl.txt", 'r') as file:
    rtl_langs = [l.strip() for l in file.read().split('\n')]


def process_one_language(lang, reverse=False):

    #load the data
    with open(f"{DATASET_PATH}/dev/{lang}.dev", 'r') as file:
        data_dev = file.read().split('\n')
    with open(f"{DATASET_PATH}/devtest/{lang}.devtest", 'r') as file:
        data_devtest = file.read().split('\n')
    data = data_dev + data_devtest

    examples = [53,140,366,504,703,779,794,871,899,936]
    data_str = " ".join(data)
    
    n_words = len(data_str.split(' '))

    dict_fertility = {"lang": lang}
    dict_len = {"lang": lang}
    dict_unknown = {"lang": lang}
    examples_tokenized = defaultdict(list)

    for tk in ALL_TOKENIZERS:
        print(f"Language {lang}: processing tokenizer {tk.pretty_name}.")
        tokens = tk.encode(data_str)
        dict_len[tk.pretty_name] = len(tokens) 
        dict_fertility[tk.pretty_name] = len(tokens) / n_words
        dict_unknown[tk.pretty_name] = tk.count_unknown(data_str)

        # process the examples:
        for i in range(len(data)):
            ex = data[i]
            processed_tokens, processed_strs = tk.align_tokens_to_text(tk.encode(ex), reverse=reverse)
            total_num_tokens = sum([len(t) for t in processed_tokens])
            tokens_per_sentence[tk.pretty_name][lang].append(total_num_tokens)

            if i in examples:
                unknown_count = tk.count_unknown(ex)
                examples_tokenized[tk.pretty_name].append({
                    "text": ex, 
                    "tokens-text": processed_strs, 
                    "tokens": processed_tokens,
                    "num_tokens": total_num_tokens,
                    "unknown_fraction": unknown_count / total_num_tokens,
                    })
            

    # save the examples:
    os.makedirs("assets/examples", exist_ok=True)
    df=pandas.DataFrame(examples_tokenized)
    # print(df)
    df.to_json(f"assets/examples/{lang}.json", force_ascii=False, indent=2)

    return dict_len, dict_unknown, dict_fertility

processed_dicts_list = []
for l in tqdm.tqdm(langs):
    processed_dicts_list.append(process_one_language(l))

# replace language code with language full name
language_map = pandas.read_csv("flores_language_map.csv", index_col=1, skipinitialspace=True)
language_map["Language"] = language_map["Language"].str.strip()

# save the raw tokenization lengths
df = pandas.DataFrame([d[0] for d in processed_dicts_list]).set_index("lang")
df = pandas.merge(df, language_map, left_index=True, right_index=True)
df.set_index("Language", inplace=True)
df.to_csv("assets/tokenization_lengths.csv")

# save the raw numbers of unknown tokens
df_unknown = pandas.DataFrame([d[1] for d in processed_dicts_list]).set_index("lang")
df_unknown = pandas.merge(df_unknown, language_map, left_index=True, right_index=True)
df_unknown.set_index("Language", inplace=True)
df_unknown.to_csv("assets/tokenization_unknown.csv")

# save the fertility rate
df_fert = pandas.DataFrame([d[2] for d in processed_dicts_list]).set_index("lang")
df_fert = pandas.merge(df_fert, language_map, left_index=True, right_index=True)
df_fert.set_index("Language", inplace=True)
df_fert.to_csv("assets/fertility.csv")

# save the fraction of unknown tokens
assert((df.columns == df_unknown.columns).all())
assert((df.index == df_unknown.index).all())
df_unknown_fraction = df_unknown.copy()
for col in df.columns:
    df_unknown_fraction[col] /= df[col]
df_unknown_fraction.to_csv("assets/tokenization_unknown_fraction.csv")

#### PLOT UNKNOWN FRACTIONS
# Extract the schemes from the column names, ignoring the "Language" column
schemes = df_unknown_fraction.columns

# Setting up the figure for the plots
num_schemes = len(schemes)
fig, axes = plt.subplots(nrows=num_schemes//3 + int(num_schemes%3 > 0), ncols=3, figsize=(18, num_schemes*2), constrained_layout=True)
axes = axes.flatten()

for i, scheme in enumerate(schemes):
    # Plotting
    axes[i].bar(df_unknown_fraction.index, df_unknown_fraction[scheme])
    axes[i].set_title(scheme)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].set_ylabel('Fraction of Unknowns')
    axes[i].set_ylim(0, df_unknown_fraction[schemes].max().max() + 0.0001)

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.savefig('./assets/unknowns.png', dpi=300, bbox_inches='tight')


# NaN the rows for which we have too many unknown tokens
THRESHOLD_FOR_TOO_MANY_UNKNOWN = 0.1
df[df_unknown_fraction > THRESHOLD_FOR_TOO_MANY_UNKNOWN] = "–––"
df.to_csv("assets/tokenization_lengths_validated.csv")

### Compute parity scores; iterate over jsons in assets/examples
def compute_parity(n_tokens, n_tokens2):
    n_tokens, n_tokens2 = np.array(n_tokens[:-1]), np.array(n_tokens2[:-1])
    ratios = ( np.abs(n_tokens) + 0.0000001 ) / ( np.abs(n_tokens2) + 0.0000001 )
    return np.mean(ratios)

#print(tokens_per_sentence)
parity_results = {}
for tokenizer_name, dict_langs in tokens_per_sentence.items():
    parity_results[tokenizer_name] = {}
    print("Computing parity for {}".format(tokenizer_name))
    for lang, n_tokens in dict_langs.items():
        parity_results[tokenizer_name][lang] = {}
        for lang2, n_tokens2 in dict_langs.items():
            parity_results[tokenizer_name][lang][lang2] = compute_parity(n_tokens, n_tokens2)

for tk_name, parity in parity_results.items():
    matrix_tk = np.array([list(x.values()) for _, x in parity.items()])
    labels = list(parity.keys())
    # Plot the heatmap with labels
    plt.figure(figsize=(10, 8))
    plt.title('Parity for tokenizer: {}'.format(tk_name))
    sns.heatmap(matrix_tk, annot=True, cmap='viridis', xticklabels=labels, yticklabels=labels)
    # Save the figure
    plt.savefig('./assets/parity_{}.png'.format(tk_name), dpi=300, bbox_inches='tight')

print(parity_results)