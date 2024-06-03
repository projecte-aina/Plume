import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_confidence(output_tokens, layer, att_head, index_batch, start_index, end_index):
    concat_matrix = None
    for attention in output_tokens.attentions[1:]:
        val = attention[layer][index_batch, att_head, :, start_index:end_index]
        if concat_matrix is None:
            concat_matrix = val
        else:
            concat_matrix = torch.concat([concat_matrix, val], dim=1)
    confidence = torch.mean( concat_matrix.max(dim=1)[0] )
    return confidence

def get_coverage(output_tokens, layer, att_head, index_batch, start_index, end_index):
    coverage = 0
    #src_tokens = output_tokens.attentions[0][layer][:, att_head, :, :].shape[-1]
    for attention in output_tokens.attentions[1:]:
        coverage += attention[layer][index_batch, att_head, :, start_index:end_index].sum().item()**2
    return coverage

def remove_after_newline(text):
    # Find the index of the first newline character
    index = text.find('\n')
    if index != -1:
        return text[:index]
    return text

def get_att_matrix(output_tokens, layer, att_head, index_batch):

    concat_matrix = output_tokens.attentions[0][layer][index_batch, att_head, :,].cpu().numpy()
    new_tokens_n = len(output_tokens.attentions[1:])
    pad_width = ((0, new_tokens_n), (0, new_tokens_n))
    att_matrix = np.pad(concat_matrix, pad_width=pad_width, mode='constant', constant_values=0)

    begin_index = concat_matrix.shape[0]
    for index, attention in enumerate(output_tokens.attentions[1:]):
        val = attention[layer][index_batch, att_head, :,].cpu().numpy()
        pad_width = ((0, 0), (0, new_tokens_n - (index + 1) ))
        row = np.pad(val, pad_width=pad_width, mode='constant', constant_values=0)
        att_matrix[begin_index] = row
        begin_index += 1
    return att_matrix

def plot_matrix(matrix, translations, save_dir):

    print(len(translations))
    plt.figure(figsize=(20, 20))

    sns.heatmap(matrix, xticklabels=translations[:-1], yticklabels=translations[:-1], cmap='viridis', cbar_kws={'label': 'Attention Weights'})
    plt.title('Attention Matrix Heatmap')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.savefig(save_dir)
    plt.close()