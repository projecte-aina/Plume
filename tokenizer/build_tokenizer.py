import argparse
from tokenizers.trainers import UnigramTrainer, BpeTrainer
from tokenizers.models import Unigram, BPE
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors, Regex
import glob
import os
import json
from transformers import PreTrainedTokenizerFast

# Initialize the parser
parser = argparse.ArgumentParser(description='Train a tokenizer model.')

# Adding arguments
parser.add_argument('--vocab_size', type=int, default=49500, help='Vocabulary size')
parser.add_argument("--files_directory", type=str, help="Files directory.")
parser.add_argument("--output", type=str, help="Output path to save the tokenizer.")
parser.add_argument("--tokenizer_type", type=str, choices=['bpe', 'unigram'], default="unigram", help="Type of tokenizer to use.")

# Parse the arguments
args = parser.parse_args()

# Select tokenizer type based on parser
if args.tokenizer_type == 'unigram':
    tokenizer = Tokenizer(Unigram())

    tokenizer.normalizer = normalizers.Sequence(
                                                [
                                                    normalizers.Replace("``", '"'),
                                                    normalizers.Replace("''", '"'),
                                                    normalizers.NFKD(),
                                                    #normalizers.StripAccents(),
                                                    normalizers.Replace(Regex(" {2,}"), " ")
                                                ]
                                            )

    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="_", add_prefix_space=True)
    tokenizer.decoder = decoders.Metaspace(replacement="▁", add_prefix_space=True)

elif args.tokenizer_type == 'bpe':
    tokenizer = Tokenizer(BPE())

    tokenizer.normalizer = normalizers.Sequence(
                                                [
                                                    normalizers.Replace("``", '"'),
                                                    normalizers.Replace("''", '"'),
                                                    normalizers.NFKD(),
                                                    #normalizers.StripAccents(),
                                                    normalizers.Replace(Regex(" {2,}"), " ")
                                                ]
                                            )
                                            
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()

else:
    raise ValueError("Unsupported tokenizer type. Use 'bpe' or 'unigram'.")


special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
language_tokens = ["[deu_Latn]", "[eng_Latn]", "[eus_Latn]", "[fra_Latn]", 
                   "[glg_Latn]", "[ita_Latn]", "[por_Latn]", "[spa_Latn]", 
                   "[cat_Latn]"]

special_tokens = special_tokens + language_tokens
train_files = glob.glob( args.files_directory + '*.txt')

# Select tokenizer type based on parser
if args.tokenizer_type == 'unigram':
    trainer = UnigramTrainer(
        vocab_size=args.vocab_size,
        show_progress=True,
        unk_token="<unk>",
        special_tokens=special_tokens
        #shrink_factor=0.75,
        #n_sub_iterations=2
        )
elif args.tokenizer_type == 'bpe':
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        show_progress=True,
        #min_frequency=2,
        special_tokens=special_tokens
    )

tokenizer_output = "{}/tokenizer.json".format(args.output)
vocab_output = "{}/vocab.json".format(args.output)
merges_output = "{}/merges.json".format(args.output)

tokenizer_fast_output = "{}/tokenizer_fast/".format(args.output)
os.makedirs(tokenizer_fast_output, exist_ok=True)

tokenizer.train(train_files, trainer=trainer)
tokenizer.save(tokenizer_output)

# create vocab.json and merges.txt
with open(vocab_output, "w") as vocab_file:
    vocab_json = json.loads(open( tokenizer_output ).read())["model"]["vocab"]
    vocab_file.write(json.dumps(vocab_json))

# save fast tokenizer
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, model_max_length=512)
fast_tokenizer.save_pretrained(tokenizer_fast_output)