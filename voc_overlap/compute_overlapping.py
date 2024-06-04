from transformers import AutoTokenizer
import json
import argparse
from normalize import preproc

def create_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-src", "--src_lang_code", type=str, help="Source language code")
	parser.add_argument("-tgt", "--tgt_lang_code", type=str, help="Target language code")
	parser.add_argument("-t", "--tokenizer", type=str, help="Path to the tokenizer")
	parser.add_argument("-f", "--flores", type=str, help="Path to flores")
	return parser

def generate_tokenized_sentence(text, max_length=5000):
	input_ids = tokenizer(text, return_tensors='pt')
	return input_ids["input_ids"]

def get_vocab(path_tokenizer):
	with open(path_tokenizer + "/tokenizer.json") as f:
		data = json.load(f)
		vocab = data["model"]["vocab"]
	return vocab

def flores_to_set_ids(flores):
	with open(flores, 'r', encoding='utf-8') as f:
		set_ids = set()
		sentences = [preproc(line) for line in f]
		for sentence in sentences:
			tokenized_text_tensor = generate_tokenized_sentence(sentence)
			tokenized_text_array = tokenized_text_tensor.numpy()
			tokenized_text_list = tokenized_text_array.tolist()[0]
			set_ids.update(tokenized_text_list)
	return set_ids

def compute_overlap_score(ids_src, ids_tgt, len_vocab):
	intersection_list = list(ids_src.intersection(ids_tgt))
	value_overlap_tgt = (len(intersection_list)/len(ids_tgt)) * 100
	return value_overlap_tgt

parser = create_parser()
args = parser.parse_args()

lang = {"deu": "de", "eus": "eu", "fra": "fr", "ita": "it", "glg": "gl", "spa": "es", "por": "pt", "eng": "en"}

tokenizer_kwargs = {
		"unk_token": '<unk>',
		"pad_token": '<pad>',
		"model_max_length": 2048,
		"trust_remote_code": True
	}

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, **tokenizer_kwargs)

vocab = get_vocab(args.tokenizer)
len_vocab = len(vocab)

flores_src = f"{args.flores}/{args.src_lang_code}_Latn.devtest"
flores_tgt = f"{args.flores}/{args.tgt_lang_code}_Latn.devtest"

ids_src = flores_to_set_ids(flores_src)
ids_tgt = flores_to_set_ids(flores_tgt)

value_overlap_tgt = compute_overlap_score(ids_src, ids_tgt, len_vocab)
print(f"{lang[args.src_lang_code]}-{lang[args.tgt_lang_code]}\t{round(value_overlap_tgt,3)}") 