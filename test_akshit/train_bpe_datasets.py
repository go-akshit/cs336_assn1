import sys
sys.path.insert(0, '../../cs336_assn1/cs336_basics')
from BPE_Tokenizer import bpe_train, BPE_Tokenizer, Tokenizer

#input_path = '/data/TinyStoriesV2-GPT4-train.txt'
#vocab_size = 10000
#special_tokens=["<|endoftext|>"]

input_path = '/data/owt_train.txt'
vocab_size = 32000
special_tokens=["<|endoftext|>"]

bpe_train(input_path, vocab_size, special_tokens)
with open('Section_2.5_bpe_owt_done', 'w') as out:
    out.write('done\n ')
