import ast
import sys
sys.path.insert(0, '../../cs336_assn1/cs336_basics')
from BPE_Tokenizer import bpe_train, BPE_Tokenizer, Tokenizer

def get_vocab_merges(vocab_path, merges_path):
    
    vocab = {}
    merges = []
    with open(vocab_path, 'r') as file:
        line = file.readline()
        while line:
            #import pdb; pdb.set_trace()
            line_split = line.split(' ', 1)
            vocab[ast.literal_eval(line_split[0])] = ast.literal_eval(line_split[1])
            line = file.readline()
    
    with open(merges_path, 'r') as f:
        line = f.readline()
        while line:
            evaluated_tuple = ast.literal_eval(line)
            merges.append(evaluated_tuple)
            line = f.readline()
    return vocab, merges

vocab_path = 'TinyStoriesV2-GPT4-train.txt_my_vocab.json'
merges_path = 'TinyStoriesV2-GPT4-train.txt_my_merges.txt'

vocab, merges = get_vocab_merges(vocab_path, merges_path)

special_tokens=["<|endoftext|>"]
input_path = '/data/TinyStoriesV2-GPT4-train.txt'
tokenizer = Tokenizer(vocab, merges, special_tokens)
test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
import pdb; pdb.set_trace()
ids = tokenizer.encode(test_string)
assert tokenizer.decode(ids) == test_string