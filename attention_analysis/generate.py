from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
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

batch_size = 1
N_ATT_HEADS = 8
N_LAYERS = 18
MAX_SENTENCES = 2000

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

    CONFIDENCE_sentence = np.zeros((N_LAYERS, N_ATT_HEADS))  # Assuming the model has 32 layers and 32 attention heads
    COVERAGE_sentence = np.zeros((N_LAYERS, N_ATT_HEADS))

    CONFIDENCE_bos = np.zeros((N_LAYERS, N_ATT_HEADS))
    COVERAGE_bos = np.zeros((N_LAYERS, N_ATT_HEADS))

    CONFIDENCE_source = np.zeros((N_LAYERS, N_ATT_HEADS))
    COVERAGE_source = np.zeros((N_LAYERS, N_ATT_HEADS))

    CONFIDENCE_target = np.zeros((N_LAYERS, N_ATT_HEADS))
    COVERAGE_target = np.zeros((N_LAYERS, N_ATT_HEADS))
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
            
            # confidence
            confidence_matrix_bos = np.zeros((N_LAYERS, N_ATT_HEADS))
            confidence_matrix_sentence = np.zeros((N_LAYERS, N_ATT_HEADS))
            confidence_matrix_source = np.zeros((N_LAYERS, N_ATT_HEADS))
            confidence_matrix_target = np.zeros((N_LAYERS, N_ATT_HEADS))
            # coverage
            coverage_matrix_bos   = np.zeros((N_LAYERS, N_ATT_HEADS))
            coverage_matrix_sentence   = np.zeros((N_LAYERS, N_ATT_HEADS))
            coverage_matrix_source   = np.zeros((N_LAYERS, N_ATT_HEADS))
            coverage_matrix_target   = np.zeros((N_LAYERS, N_ATT_HEADS))
            
            for layer in range(N_LAYERS):
                for head in range(N_ATT_HEADS):
                    for index_in_batch in range(batch_size):
                        start = starts[index_in_batch]
                        end = ends[index_in_batch]
                        #print(end)
                        #print(start)
                        zero_index, start_index, end_index, END = get_sentence_token_indices(tokenizer, start, batched_prompt, end, inputs, index_in_batch)
                        #print(zero_index, start_index, end_index, END)
                        confidence_matrix_bos[layer, head] = get_confidence(translated_tokens, layer, head, index_in_batch, 0, start_index-1)
                        confidence_matrix_source[layer, head] = get_confidence(translated_tokens, layer, head, index_in_batch, start_index-1, start_index)
                        confidence_matrix_sentence[layer, head] = get_confidence(translated_tokens, layer, head, index_in_batch, start_index, end_index)
                        confidence_matrix_target[layer, head] = get_confidence(translated_tokens, layer, head, index_in_batch, end_index, END)
                        
                        coverage_matrix_bos[layer, head] = get_coverage(translated_tokens, layer, head, index_in_batch, 0, start_index-1)
                        coverage_matrix_source[layer, head] = get_coverage(translated_tokens, layer, head, index_in_batch, start_index-1, start_index)
                        coverage_matrix_sentence[layer, head] = get_coverage(translated_tokens, layer, head, index_in_batch, start_index, end_index)
                        coverage_matrix_target[layer, head] = get_coverage(translated_tokens, layer, head, index_in_batch, end_index, END)

            CONFIDENCE_bos += confidence_matrix_bos
            COVERAGE_bos   += coverage_matrix_bos

            CONFIDENCE_sentence += confidence_matrix_sentence
            COVERAGE_sentence   += coverage_matrix_sentence
            
            CONFIDENCE_source += confidence_matrix_source
            COVERAGE_source   += coverage_matrix_source
            
            CONFIDENCE_target += confidence_matrix_target
            COVERAGE_target   += coverage_matrix_target
            
            N += batch_size

            # Decode
            input_token_length = inputs["input_ids"].size(1)
            generated_tokens = translated_tokens.sequences
            #print(inputs["input_ids"].size())
            #print(input_token_length)
            #print(generated_tokens.shape)
            translation = tokenizer.decode(generated_tokens[0, input_token_length:], skip_special_tokens=True)
            print(translation)
            for skip_sent in to_skip:
                if not skip_sent:
                    o.write( remove_after_newline(translation).strip() + "\n")
                else:
                    o.write("\n")
            if N > MAX_SENTENCES:
                break

    CONFIDENCE_bos /= N
    COVERAGE_bos   /= N

    CONFIDENCE_sentence /= N
    COVERAGE_sentence   /= N

    CONFIDENCE_source /= N
    COVERAGE_source   /= N

    CONFIDENCE_target /= N
    COVERAGE_target   /= N
    print('Translation completed')

    np.save(f'{args.output}/confidence_sentence.npy', CONFIDENCE_sentence)
    np.save(f'{args.output}/confidence_source.npy', CONFIDENCE_source)
    np.save(f'{args.output}/confidence_target.npy', CONFIDENCE_target)
    np.save(f'{args.output}/confidence_bos.npy', CONFIDENCE_bos)

    np.save(f'{args.output}/coverage_sentence.npy', COVERAGE_sentence)
    np.save(f'{args.output}/coverage_source.npy', COVERAGE_source)
    np.save(f'{args.output}/coverage_target.npy', COVERAGE_target)
    np.save(f'{args.output}/coverage_bos.npy', COVERAGE_bos)

    CONFIDENCE_sentence_n = normalize_matrix(CONFIDENCE_sentence)
    COVERAGE_sentence_n = normalize_matrix(COVERAGE_sentence)

    CONFIDENCE_source_n = normalize_matrix(CONFIDENCE_source)
    COVERAGE_source_n = normalize_matrix(COVERAGE_source)

    CONFIDENCE_target_n = normalize_matrix(CONFIDENCE_target)
    COVERAGE_target_n = normalize_matrix(COVERAGE_target)

    CONFIDENCE_bos_n = normalize_matrix(CONFIDENCE_bos)
    COVERAGE_bos_n = normalize_matrix(COVERAGE_bos)

    np.save(f'{args.output}/confidence_sentence_normalized.npy', CONFIDENCE_sentence_n)
    np.save(f'{args.output}/confidence_source_normalized.npy', CONFIDENCE_source_n)
    np.save(f'{args.output}/confidence_target_normalized.npy', CONFIDENCE_target_n)
    np.save(f'{args.output}/confidence_bos_normalized.npy', CONFIDENCE_bos_n)

    np.save(f'{args.output}/coverage_sentence_normalized.npy', COVERAGE_sentence_n)
    np.save(f'{args.output}/coverage_source_normalized.npy', COVERAGE_source_n)
    np.save(f'{args.output}/coverage_target_normalized.npy', COVERAGE_target_n)
    np.save(f'{args.output}/coverage_bos_normalized.npy', COVERAGE_bos_n)

if __name__ == "__main__":
    main()