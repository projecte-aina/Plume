from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os 
import numpy as np
import argparse
from utils import *
from normalize import preproc
import unicodedata

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-src", "--src_lang_code", type=str, default='spa_Latn', help="Source language code")
    parser.add_argument("-tgt", "--tgt_lang_code", type=str, default='quy_Latn', help="Target language code")
    parser.add_argument("-m", "--model", type=str, help="Path to the pretrained model")
    parser.add_argument("-o", "--output", type=str, help="Output path to store analysis")
    parser.add_argument("-d", "--data", type=str, help="Path to file to translate")
    parser.add_argument('--float16', action='store_false', help='Use float16')
    return parser

def get_sentence_token_indices(tokenizer, start, batched_prompt, end, inputs, index_in_batch):
    tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in inputs['input_ids']]
    sent = batched_prompt[index_in_batch]
    #print(tokens)
    start_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in tokenizer(start, return_tensors="pt")['input_ids'] ][0]
    #print(start_tokens)
    start_index = None
    for j in range(len(tokens[index_in_batch]) - len(start_tokens) + 1):
        if tokens[index_in_batch][j:j+len(start_tokens)] == start_tokens:
            start_index = j + len(start_tokens)
    
    end_tokens = [ tokenizer.convert_ids_to_tokens(ids) for ids in tokenizer(end, return_tensors="pt")['input_ids'] ][0]
    #print(end_tokens)
    end_index = None
    for j in range(len(tokens[index_in_batch]) - len(end_tokens) + 1):
        if tokens[index_in_batch][j:j+len(end_tokens)] == end_tokens:
            end_index = j

    END = end_index + len(end_tokens)
    zero_index = start_index - len(start_tokens)
    return zero_index, start_index, end_index, END 

def normalize_matrix(matrix):
    mean = np.mean(matrix)
    std_dev = np.std(matrix)
    normalized_matrix = (matrix - mean) / std_dev
    return normalized_matrix

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

batch_size = 1
N_ATT_HEADS = 8
N_LAYERS = 18
MAX_SENTENCES = 2
def main():
    parser = create_parser()
    args = parser.parse_args()

    path_translation = f'{args.output}/translation.txt'
    tokenizer_kwargs = {
        "unk_token": '<unk>',
        "pad_token": '<pad>',
        "model_max_length": 2048,
        "trust_remote_code": True
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
    if args.float16:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, 
                                                     trust_remote_code=True, output_attentions=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, output_attentions=True)

    model.cuda()
    print('Model loaded')
    print(model.device)    

    N = 0
    
    with open(args.data, 'r', encoding='utf-8') as f, open(path_translation, 'w', encoding='utf-8') as o:
        sentences = [preproc(line) for line in f]
        for batched_input in batch(sentences, batch_size):

            batched_prompt = []
            starts = []
            ends = []
            to_skip = []
            for sent in batched_input:
                prompt = '<s> [{}] {} \n[{}]'.format(args.src_lang_code, sent, args.tgt_lang_code)
                #prompt = '<s> <s> {} \n[{}]'.format(sent, args.tgt_lang_code)
                #f'{src_name}: {sent}\n{tgt_name}:'
                start = '<s> [{}]'.format(args.src_lang_code) 
                # f'{src_name}: '
                end   = ' \n[{}]'.format(args.tgt_lang_code)
                #f'\n{tgt_name}:'
                if sent.strip() == '':
                    # create a list for down skip if this
                    to_skip.append(True)
                    continue
                batched_prompt.append( prompt )
                starts.append(start)
                ends.append(end)
                to_skip.append(False)

            inputs = tokenizer(batched_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512, return_token_type_ids=False)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            #print(inputs)
            translated_tokens = model.generate(**inputs, max_length=512, return_dict_in_generate=True, output_scores=False, output_attentions=True)

            # Decode
            input_token_length = inputs["input_ids"].size(1)
            generated_tokens = translated_tokens.sequences
            
            translation = tokenizer.decode(generated_tokens[0,:], skip_special_tokens=True)
            translations = [tokenizer.decode(generated_tokens[0, i], skip_special_tokens=False) for i in range(generated_tokens[0,:].size(0))]
            print(translations)

            for layer in range(N_LAYERS):
                for head in range(N_ATT_HEADS):
                    for index_in_batch in range(batch_size):
                        att_matrix_sentence_n = get_att_matrix(translated_tokens, layer, head, index_in_batch)
                        save_dir = f'{args.output}/att_matrices/{layer}/{head}/'
                        ensure_directory_exists(save_dir)
                        np.save(f'{save_dir}/att.matrix_{N+index_in_batch}.npy', att_matrix_sentence_n)
                        plot_matrix(att_matrix_sentence_n, translations, f'{save_dir}/att.matrix_{N+index_in_batch}.png')
            
            N += batch_size

            print(translation)
            break

    #print('Translation completed')
    #np.save(f'{args.output}/att.matrix.npy', att_matrix_sentence_n)

if __name__ == "__main__":
    main()