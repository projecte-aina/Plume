from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
import torch
import argparse
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from transformers import CONFIG_MAPPING
from normalize import preproc
import unicodedata

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-src", "--src_lang_code", type=str, default='spa_Latn', help="Source language code")
    parser.add_argument("-tgt", "--tgt_lang_code", type=str, default='quy_Latn', help="Target language code")
    parser.add_argument("-voc", "--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("-m", "--model", type=str, help="Path to the pretrained model")
    parser.add_argument("-o", "--output", type=str, help="Output path to store translations")
    parser.add_argument("-d", "--data", type=str, help="Path to file to translate")
    parser.add_argument("-b", "--beam", default=1, type=int, help="Beam size")
    parser.add_argument("-ch", "--checkpoint", default='deepspeed', type=str, help="Checkpoint type to use")
    parser.add_argument("-pen", "--penalty", default=1.0, type=float, help="Penalty")
    parser.add_argument('--float16', type=str, help='Use float16')
    parser.add_argument('--ignore_source', type=str, help='Do not use source token')
    parser.add_argument("-preproc", "--preproc", default='yes', type=str, help="Whether or not to use NLLB preprocessing")
    return parser

parser = create_parser()
args = parser.parse_args()

tokenizer_kwargs = {
        "unk_token": '<unk>',
        "pad_token": '<pad>',
        "model_max_length": 2048,
        "trust_remote_code": True
    }

tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
print('Tokenizer loaded')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print('Loading model')

if args.checkpoint == 'deepspeed':
    config_overrides = "vocab_size={},attention_bias=false,attention_dropout=0.0,head_dim=256,hidden_size=2048,initializer_range=0.02,intermediate_size=16384,max_position_embeddings=8192,num_attention_heads=8,num_hidden_layers=18,num_key_value_heads=1,rms_norm_eps=1e-06,rope_theta=10000.0,pad_token_id=3,bos_token_id=0,eos_token_id=1".format(args.vocab_size)
    config = CONFIG_MAPPING['gemma']()
    config.update_from_string(config_overrides)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model = load_state_dict_from_zero_checkpoint(model, args.model)
else:
    if args.float16 == 'yes':
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model)

model.cuda()
print('Model loaded')
print(model.device)

def remove_after_newline(text):
    # Find the index of the first newline character
    index = text.find('\n')
    if index != -1:
        return text[:index]
    return text

# Function to generate text
def generate_text(prompt, max_length=512):
    # Encode input context
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    # Generate output
    output_ids = model.generate(input_ids, max_length=max_length, num_beams=args.beam, repetition_penalty=args.penalty ) # num_beams=args.beam repetition_penalty=1
    input_length = input_ids.shape[1]
    # Decode text
    generated_text = tokenizer.decode(output_ids[0, input_length:], skip_special_tokens=True)
    generated_text = remove_after_newline(generated_text)
    return generated_text.replace('\n', '').strip()

# Example usage
#input_sentence = "<s> [fra_Latn] Il a marqué 2 buts et fait 2 passes décisives durant le match contre les Atlanta Trashers, remporté 5-3 par Washington. \n[cat_Latn]"
#generated_text = generate_text(input_sentence)
#print(generated_text)
print('[INFO] Translating from {} to {}'.format(args.src_lang_code, args.tgt_lang_code))
with open(args.data, 'r', encoding='utf-8') as f, open(args.output, 'w', encoding='utf-8') as o:
    sentences = [line for line in f]
    for sentence in sentences:
        if args.preproc:
            sentence = preproc(sentence)

        if args.ignore_source == 'yes':
            input_sentence = '<s> <s> {} \n[{}]'.format(sentence, args.tgt_lang_code)
            print( '[INFO] Ignoring source language tag. Input Sentence: {}'.format(input_sentence) )
        else:
            input_sentence = '<s> [{}] {} \n[{}]'.format(args.src_lang_code, sentence, args.tgt_lang_code)
        generated_text = generate_text(input_sentence)
        generated_text = unicodedata.normalize("NFKC", generated_text)
        print(generated_text)
        o.write(generated_text + "\n")

print('Translation completed')