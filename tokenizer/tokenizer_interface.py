#!/usr/bin/env python
# Implementation adapted from Petrov, A., La Malfa, E., Torr, P., & Bibi, A. (2024). 
# Language model tokenizers introduce unfairness between languages. Advances in 
# Neural Information Processing Systems, 36.
# https://github.com/AleksandarPetrov/tokenization-fairness

from abc import ABC, abstractmethod, abstractproperty
import os
from typing import List, Union
from itertools import cycle
from tokenizers import Tokenizer
from transformers import AutoTokenizer
import torch

# abstract class for tokenizers inheriting from ABC
class TokenizerInterface(ABC):

    NOT_COMPLETE_SYMBOL_ORD = None

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, text: List[int]) -> str:
        raise NotImplementedError

    @abstractproperty
    def pretty_name(self) -> str:
        raise NotImplementedError
    
    @classmethod
    def format_color(cls, text, color):
        """
        Prints the specified text in the specified color.
        """
        colors = {
            "black": "\u001b[40m",
            "red": "\u001b[41m",
            "green": "\u001b[42m",
            "yellow": "\u001b[43m",
            "blue": "\u001b[44m",
            "magenta": "\u001b[45m",
            "cyan": "\u001b[46m",
            "white": "\u001b[47m",
            "reset": "\033[0m",
        }
        if color not in colors:
            raise ValueError("Invalid color: {}".format(color))
        return colors[color] + text + colors["reset"]

    def print_pretty_tokens(self, tokens: List[int], print_total=False):

        token_words = [self.decode([t]) for t in tokens ]
        colors = ["red", "green", "blue", "magenta", "cyan"]
        
        for t, w, c in zip(tokens, token_words, cycle(colors)):
            print(self.format_color(str(t).ljust(max(len(str(t)), len(w)), '~'), c), end="")

        print("")

        for t, w, c in zip(tokens, token_words, cycle(colors)):
            print(self.format_color(str(w).ljust(max(len(str(t)), len(w)), '~'), c), end="")
        print("")
        
        if print_total:
            print(f"Total {len(tokens)} tokens")
        
    def print_pretty_text(self, text: str, print_total=False):
        tokens = self.encode(text)
        self.print_pretty_tokens(tokens, print_total)

    def print_pretty(self, test_or_tokens: Union[str, List[int]], print_total=False):
        if isinstance(test_or_tokens, str):
            self.print_pretty_text(test_or_tokens, print_total=print_total)
        elif isinstance(test_or_tokens, list):
            self.print_pretty_tokens(test_or_tokens, print_total=print_total)
        else:
            raise ValueError(f"Invalid input type for print_pretty. Must be str or list of ints. Found {type(test_or_tokens)}")

    def align_tokens_to_text(self, tokens, reverse=False):
        processed_tokens = []
        processed_strs = []

        pred = []
        for t in tokens:
            unicode_error = False
            dec = ""

            curr = pred + [t]

            try:
                dec = self.decode(curr)
            except UnicodeDecodeError:
                unicode_error = True

            if (len(dec) > 1) or (len(dec)==1 and ord(dec) != self.NOT_COMPLETE_SYMBOL_ORD) or unicode_error:
                processed_tokens.append(tuple(curr))
                processed_strs.append(dec)
                pred = []
            else:
                pred.append(t)

        if reverse:
            processed_tokens = processed_tokens[::-1]
            processed_strs = processed_strs[::-1]

        return processed_tokens, processed_strs

    def latex_pretty(self, text, font="", reverse=False):
        tokens = self.encode(text)
        processed_tokens = []
        processed_strs = []
        wrapword_command = []

        pred = []
        for t in tokens:
            curr = pred + [t]
            dec = self.decode(curr)
            if (len(dec) > 1) or (len(dec)==1 and ord(dec) != self.NOT_COMPLETE_SYMBOL_ORD):
                processed_tokens.append(tuple(curr))
                processed_strs.append(dec)
                if len(curr) == 1:
                    wrapword_command.append(["wrapword"])
                elif len(curr) == 2:
                    wrapword_command.append(["wrapwordleft", "wrapwordright"])
                else:
                    wrapword_command.append(["wrapwordleft"] + ["wrapwordcenter"] * (len(curr)-2) + ["wrapwordright"])
                pred = []
            else:
                pred.append(t)

        if reverse:
            processed_tokens = processed_tokens[::-1]
            processed_strs = processed_strs[::-1]
            wrapword_command = wrapword_command[::-1]

        prefix = """
        \\begin{center}
        \\begingroup
        \\setlength{\\tabcolsep}{2pt}
        \\renewcommand{\\arraystretch}{0}
        \\begin{tabular}{
        """+ "c" * len(processed_tokens) + "}\n"

        codes = " & ".join(["".join(["\\"+ww+"{"+str(t)+"}" for ww, t in zip(ww_tup, token_tup)]) for ww_tup, token_tup in zip(wrapword_command, processed_tokens)]) + "\\\\\n"
        words = " & ".join(["\\wrapword{"+font + s +"}"  for s in processed_strs]) + "\n"

        suffix = """
        \\end{tabular}
        \\endgroup
        \\end{center}
        """

        return prefix + codes + words + suffix

    @abstractmethod
    def count_unknown(self, text: str) -> int:
        raise NotImplementedError


class HuggingFaceTokenizer(TokenizerInterface):
    NOT_COMPLETE_SYMBOL_ORD = 65533
    init_kwargs = {}

    def __init__(self, path, type_alg, sampling, size, vocab_size):
        #self.encoder = AutoTokenizer.from_pretrained(self.tokenizer, **self.init_kwargs)
        self.tokenizer_name = "{}.{}_{}M_{}".format(type_alg, sampling, size, vocab_size)
        self.encoder = AutoTokenizer.from_pretrained(path, unk_token = '<unk>')

    def encode(self, text: str) -> List[int]:
        return self.encoder.convert_tokens_to_ids(self.encoder.tokenize(text))
    
    def decode(self, tokens: List[int]) -> str:
        return self.encoder.decode(tokens)

    @property
    def pretty_name(self) -> str:
        return self.tokenizer_name
    
    def count_unknown(self, text: str) -> int:
        #print(self.encoder.unk_token)
        unknown_token = self.encoder.convert_tokens_to_ids([self.encoder.unk_token])[0]
        tokens = self.encode(text)
        tokens_wo_unk = [t for t in tokens if t != unknown_token]
        text_wo_unk = self.decode(tokens_wo_unk)
        return max(0, int(len(tokens)*(len(text)-len(text_wo_unk))/len(text)))