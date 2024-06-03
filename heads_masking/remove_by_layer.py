from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import os
import numpy as np
from normalize import preproc
import unicodedata
import sacrebleu
import matplotlib.pyplot as plt
import pandas as pd

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-src", "--src_lang_code", type=str, default='spa_Latn', help="Source language code")
    parser.add_argument("-tgt", "--tgt_lang_code", type=str, default='quy_Latn', help="Target language code")
    parser.add_argument("-m", "--model", type=str, help="Path to the pretrained model")
    parser.add_argument("-att", "--att_matrix_dir", type=str, help="Path to the attention matrix dir")
    parser.add_argument("-d", "--data", type=str, help="Path to file to translate")
    parser.add_argument("-tgtd", "--tgt_data", type=str, help="Path to target file")
    parser.add_argument("-o", "--output", type=str, help="Output path to store translations")
    return parser

parser = create_parser()
args = parser.parse_args()

tokenizer_kwargs = {
        "unk_token": '<unk>',
        "pad_token": '<pad>',
        "model_max_length": 2048,
        "trust_remote_code": True
    }

print(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
print('Tokenizer loaded')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print('Loading model')

model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
model.cuda()
print('Model loaded')
print(model.device)

def remove_after_newline(text):
    # Find the index of the first newline character
    index = text.find('\n')
    if index != -1:
        return text[:index]
    return text

def generate_text_with_masked_heads( prompt, heads_to_mask, max_length=512 ):
    """
    Generates text from a given prompt with specified attention heads masked.

    Args:
    - prompt (str): Initial text to start generation.
    - heads_to_mask (list of tuples): List of (layer_index, head_index) to be masked.
    - max_length (int): Maximum length of the generated text sequence.

    Returns:
    - str: Generated text.
    """

    # Convert prompt to input_ids
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    input_length = input_ids.shape[1]
    eos_token_id = 1

    # Initialize past_key_values
    past_key_values = None
    output_sequence = input_ids
    # Generate tokens
    for _ in range(max_length):
        outputs = model(input_ids, past_key_values=past_key_values)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

        # Check if the next token is the end-of-sentence token
        if next_token == eos_token_id:
            output_sequence = torch.cat([output_sequence, next_token], dim=-1)
            break

        output_sequence = torch.cat([output_sequence, next_token], dim=-1)
        input_ids = next_token

        # Update and modify past_key_values
        past_key_values = []
        for i, (k, v) in enumerate(outputs.past_key_values):
            # Check if the current layer needs head masking
            for (layer_idx, head_idx) in heads_to_mask:
                if i == layer_idx:
                    # Mask the specific head in both key and value tensors
                    k[:, :, head_idx, :] = 0
                    v[:, :, head_idx, :] = 0
            past_key_values.append((k, v))

    generated_text = tokenizer.decode(output_sequence[0][input_length:], skip_special_tokens=True)
    generated_text = remove_after_newline(generated_text)
    return generated_text
    
print('Using as metric: {}'.format(metric_filter))
metric = np.load(os.path.join(args.att_matrix_dir, 'coverage_bos.npy')).mean(axis=1) + np.load(os.path.join(args.att_matrix_dir, 'coverage_target.npy')).mean(axis=1) + np.load(os.path.join(args.att_matrix_dir, 'coverage_source.npy')).mean(axis=1) + np.load(os.path.join(args.att_matrix_dir, 'coverage_sentence.npy')).mean(axis=1)
min_value = metric.min()
max_value = metric.max()
metric = (metric - min_value) / (max_value - min_value)


conf_source  = np.load(os.path.join(args.att_matrix_dir, 'confidence_source.npy')).mean(axis=1)
conf_target  = np.load(os.path.join(args.att_matrix_dir, 'confidence_target.npy')).mean(axis=1)
conf_sentence = np.load(os.path.join(args.att_matrix_dir, 'confidence_sentence.npy')).mean(axis=1)
conf_bos = np.load(os.path.join(args.att_matrix_dir, 'confidence_bos.npy')).mean(axis=1)

cov_source  = np.load(os.path.join(args.att_matrix_dir, 'coverage_source.npy')).mean(axis=1)
cov_target  = np.load(os.path.join(args.att_matrix_dir, 'coverage_target.npy')).mean(axis=1)
cov_sentence = np.load(os.path.join(args.att_matrix_dir, 'coverage_sentence.npy')).mean(axis=1)
cov_bos = np.load(os.path.join(args.att_matrix_dir, 'coverage_bos.npy')).mean(axis=1)

print(metric)
print(metric.shape)
#min_value = metric.min()
#max_value = metric.max()
#metric = (metric - min_value) / (max_value - min_value)

#sentence="Two songs from the movie, Audition (The Fools Who Dream) and City of Stars, received nominations for best original song."

MAX_SENTENCES=400
print('[INFO] Translating from {} to {}'.format(args.src_lang_code, args.tgt_lang_code))

thresholds = [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LAYERS = 18
bleus = []
n_masked_heads = []
conf_source_all  = []
conf_target_all  = []
conf_sentence_all = []
conf_bos_all = []

cov_source_all  = []
cov_target_all  = []
cov_sentence_all = []
cov_bos_all = []

heads_per_layer = { }  # Assuming 8 layers for example

#masked_heads = [ (layer, i) for i in range(8) ]
for thrs in thresholds:
    #condition = metric < thrs
    indices = np.where(metric < thrs)[0]
    print('Masking layers: {}'.format(indices))
    masked_heads = [(i, j) for i in indices for j in range(8)]

    # Count masked heads per layer
    layer_count = {i: 0 for i in range(LAYERS)}
    for layer, head in masked_heads:
        layer_count[layer] += 1

    heads_per_layer[thrs] = layer_count

    save_dir = os.path.join( args.output, 'translations_thr_{}.txt'.format(thrs) )
    with open(args.data, 'r', encoding='utf-8') as f, open(args.tgt_data, 'r', encoding='utf-8') as tgt_f, open(save_dir, 'w', encoding='utf-8') as o:
        sentences = [line for line in f]  #[:MAX_SENTENCES]
        sentences_tgt = [line for line in tgt_f]  #[:MAX_SENTENCES]
        translations = []
        for sentence in sentences:
            sentence = preproc(sentence)
            input_sentence = '<s> [{}] {} \n[{}]'.format(args.src_lang_code, sentence, args.tgt_lang_code)
            generated_text = generate_text_with_masked_heads(input_sentence, masked_heads)
            generated_text = unicodedata.normalize("NFKC", generated_text)
            #print(generated_text)
            translations.append(generated_text)
            o.write(generated_text + "\n")
        bleu = sacrebleu.corpus_bleu(translations, [sentences_tgt]).score
        bleus.append(bleu)
        n_masked_heads.append(len(masked_heads))
        conf_source_all.append(sum(conf_source[i] for i in indices))
        conf_target_all.append(sum(conf_target[i] for i in indices))
        conf_sentence_all.append(sum(conf_sentence[i] for i in indices))
        conf_bos_all.append(sum(conf_bos[i] for i in indices))

        cov_source_all.append(sum(cov_source[i] for i in indices))
        cov_target_all.append(sum(cov_target[i] for i in indices))
        cov_sentence_all.append(sum(cov_sentence[i] for i in indices))
        cov_bos_all.append(sum(cov_bos[i] for i in indices))

        print('[INFO] Threshold: {} BLEU: {}'.format( thrs, bleu ))
        print('Number of masked heads: {}'.format(len(masked_heads)))

print( heads_per_layer )

# Create DataFrame
dict_df = {
    'Thresholds': thresholds,
    'BLEUs': bleus,
    'Number of Masked Heads': n_masked_heads,
    'conf Source': conf_source_all,
    'conf Target': conf_target_all,
    'conf Sentence': conf_sentence_all,
    'conf BOS': conf_bos_all,
    'cov Source': cov_source_all,
    'cov Target': cov_target_all,
    'cov Sentence': cov_sentence_all,
    'cov BOS': cov_bos_all
}

for layer in range(LAYERS):
    dict_df[f'layer_{layer}_del_heads'] = [heads_per_layer[i][layer] for i in thresholds]

df = pd.DataFrame(dict_df)

df.to_csv( os.path.join( args.output, 'thresholds_bleus_masked_heads.csv' ), index=False)

fig, (ax1, ax3, ax4) = plt.subplots(1, 3, figsize=(13, 4))
color = 'tab:blue'
ax1.set_xlabel('Threshold')
ax1.set_ylabel('BLEU', color=color)
ax1.plot(thresholds, bleus, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel('Number of Masked Heads', color=color)
ax2.plot(thresholds, n_masked_heads, color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax3.set_xlabel('Threshold')
ax3.set_ylabel('Confidence')
ax3.plot(thresholds, conf_source_all, label='conf Source', color='tab:green')
ax3.plot(thresholds, conf_target_all, label='conf Target', color='tab:orange')
ax3.plot(thresholds, conf_sentence_all, label='conf Sentence', color='tab:purple')
ax3.plot(thresholds, conf_bos_all, label='conf BOS', color='tab:brown')
ax3.legend()

ax4.set_xlabel('Threshold')
ax4.set_ylabel('Coverage')
ax4.plot(thresholds, cov_source_all, label='cov Source', color='tab:green')
ax4.plot(thresholds, cov_target_all, label='cov Target', color='tab:orange')
ax4.plot(thresholds, cov_sentence_all, label='cov Sentence', color='tab:purple')
ax4.plot(thresholds, cov_bos_all, label='cov BOS', color='tab:brown')
ax4.legend()

if metric_filter == 'confidence_bos_inverse':
    fig.suptitle('Making heads by 1-confidence BOS')
if metric_filter == 'confidence_bos':
    fig.suptitle('Making heads by confidence BOS')
if metric_filter == 'confidence_target_inverse':
    fig.suptitle('Making heads by 1-confidence Target')
if metric_filter == 'confidence_target':
    fig.suptitle('Making heads by confidence Target')
if metric_filter == 'linear_comb':
    fig.suptitle('Making heads by Coverage BOS + SRC + Sentence + TGT')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.title('BLEU Score and Number of Masked Heads by Threshold')
plt.savefig(os.path.join( args.output, 'fig_{}.png'.format(metric_filter) ))
plt.close()