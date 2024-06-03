import argparse
import os
import numpy as np
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict
import pandas as pd
from src.distances import subspace_distance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from umap import UMAP
from umap.plot import _get_embedding
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt

CONSIDERED_LANGS = ['eng_Latn', 'fra_Latn', 'ita_Latn', 'cat_Latn', 'por_Latn', 'deu_Latn', 'eus_Latn', 'glg_Latn', 'spa_Latn']
COLUMNS_CSV = ['index_sentence', 'id', 'index_id', 'src_lang_code', 'tgt_lang_code', 'bos', 'src_tag_token', 'last_token', 'text_token']

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--total_dims', type=int, default=2048)
    return parser

# keep_types = ['first_target', 'only_tgt', 'only_src_tgt']
def get_lang_umap(args, langs, layer, n_components=2, n_neighbors=15, min_dist=0.1, keep_type = 'first_target'):
    all_lang_reps = []
    all_to_keep = []
    for lang_i, lang in enumerate(langs):
        lang_reps = np.load(os.path.join(args.output_dir, lang, "layer{}_reps.npy".format(layer)), allow_pickle=False)
        to_keep = pd.read_csv(os.path.join(args.output_dir, lang, 'input_ids.csv'), names = COLUMNS_CSV)
        
        if keep_type == 'first_target':
            # 'bos', 'src_tag_token', 'last_token', 'text_token'
            condition = ( to_keep['tgt_lang_code'] == 'glg_Latn') & (to_keep['index_sentence'] >= 0) & (to_keep['index_sentence'] < 1) & (to_keep['bos'] == 'no') & (to_keep['src_tag_token'] == 'no') & (to_keep['last_token'] == 'no') & (to_keep['text_token'] != ' [n]')
            filtered_df = to_keep[condition]
            to_keep_indices = filtered_df.index.tolist()

        lang_reps = lang_reps[to_keep_indices]
        print('Embeddings from {}: {}'.format(lang, lang_reps.shape))
        
        all_lang_reps.append(lang_reps)
        #all_lang_labels.append(to_keep[group_by].values)
        all_to_keep.append(filtered_df)

    all_lang_reps = np.concatenate(all_lang_reps, axis=0) # Concatenate across languages.
    print('Embeddings concatenated: {}'.format(all_lang_reps.shape))
    #all_lang_labels = np.concatenate(all_lang_labels, axis=0)
    #print(all_lang_labels.shape)
    concatenated_to_keep = pd.concat(all_to_keep, ignore_index=True)

    scaler = RobustScaler() #StandardScaler()
    scaled_lang_reps = scaler.fit_transform(all_lang_reps)

    umap = UMAP(n_components=n_components, n_neighbors=n_neighbors, metric='cosine', min_dist=min_dist) # , n_neighbors=n_neighbors, min_dist=min_dist, metric='euclidean'
    umap_reps = umap.fit_transform(scaled_lang_reps)
    
    np.save(os.path.join(args.output_dir, "lang_umap_{}_layer{}.npy".format(keep_type, layer)), umap_reps, allow_pickle=False)
    concatenated_to_keep.to_csv( os.path.join(args.output_dir, "info_rows_umap_{}.csv".format(keep_type) ), index=False )
    return umap_reps

def get_plot(args, keep_type, layer):
    df = pd.read_csv( os.path.join(args.output_dir, "info_rows_umap_{}.csv".format(keep_type) ) )
    matrix_path = os.path.join(args.output_dir, "lang_umap_{}_layer{}.npy".format(keep_type, layer))
    matrix = np.load(matrix_path)
    # Map each unique source language code to a unique color
    unique_lang_codes = df['src_lang_code'].unique()
    colors = plt.cm.get_cmap('tab20', len(unique_lang_codes))
    lang_to_color = {lang: colors(i) for i, lang in enumerate(unique_lang_codes)}

    # Assign colors to each row in the DataFrame based on source language code
    df['color'] = df['src_lang_code'].map(lang_to_color)

    # Create the scatter plot
    plt.figure(figsize=(7, 7))
    for lang, group in df.groupby('src_lang_code'):
        idx = group.index
        plt.scatter(matrix[idx, 0], matrix[idx, 1], s=10, color=group['color'].iloc[0], label=lang, alpha=0.7)

    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title('Scatter Plot of Languages')
    plt.legend(title='Source Language Code', loc='upper left', bbox_to_anchor=(1, 1) )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join( args.output_dir, 'umap_layer_{}.png'.format(layer) ), dpi = 300)
    plt.close()

def get_lang_umap_voronoi(args, langs, layer, n_components=2, n_neighbors=15, min_dist=0.1, keep_type = 'first_target'):
    all_lang_reps = []
    all_to_keep = []
    for lang_i, lang in enumerate(langs):
        lang_reps = np.load(os.path.join(args.output_dir, lang, "layer{}_reps.npy".format(layer)), allow_pickle=False)
        to_keep = pd.read_csv(os.path.join(args.output_dir, lang, 'input_ids.csv'), names = COLUMNS_CSV)
        
        if keep_type == 'first_target':
            condition = ( to_keep['tgt_lang_code'] == 'glg_Latn') & (to_keep['index_sentence'] >= 0) & (to_keep['index_sentence'] < 1) & (to_keep['bos'] == 'no') & (to_keep['src_tag_token'] == 'no') & (to_keep['last_token'] == 'no') & (to_keep['text_token'] != ' [n]')
            filtered_df = to_keep[condition]
            to_keep_indices = filtered_df.index.tolist()

        lang_reps = lang_reps[to_keep_indices]
        print('Embeddings from {}: {}'.format(lang, lang_reps.shape))
        
        all_lang_reps.append(lang_reps)
        #all_lang_labels.append(to_keep[group_by].values)
        all_to_keep.append(filtered_df)

    all_lang_reps = np.concatenate(all_lang_reps, axis=0) # Concatenate across languages.
    print('Embeddings concatenated: {}'.format(all_lang_reps.shape))
    #all_lang_labels = np.concatenate(all_lang_labels, axis=0)
    #print(all_lang_labels.shape)
    concatenated_to_keep = pd.concat(all_to_keep, ignore_index=True)

    #scaler = RobustScaler() #StandardScaler()
    #scaled_lang_reps = scaler.fit_transform(all_lang_reps)

    umap = UMAP(n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=0,
                metric='cosine',
                output_metric='haversine',
                random_state=1
                ).fit(all_lang_reps)
    print('[INFO] Fit done!')
    umap_reps = _get_embedding(umap)
    
    np.save(os.path.join(args.output_dir, "lang_umap_voronoi_{}_layer{}.npy".format(keep_type, layer)), umap_reps, allow_pickle=False)
    concatenated_to_keep.to_csv( os.path.join(args.output_dir, "info_rows_umap_voronoi_{}.csv".format(keep_type) ), index=False )
    return umap_reps

def main(args):
    for layer in range(0, 19):
        get_lang_umap(args, CONSIDERED_LANGS, layer, n_neighbors=9, min_dist=0.5)
        get_lang_umap_voronoi(args, CONSIDERED_LANGS, layer, n_neighbors=8)
        get_plot(args, 'first_target', layer )
    
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)