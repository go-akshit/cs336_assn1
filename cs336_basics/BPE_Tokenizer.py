import regex as re
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class BPE_Tokenizer:
    vocab: Dict[int, bytes]
    merges: List[Tuple[bytes, bytes]]

def bpe_train(input_path: str, vocab_size: int, special_tokens: list[str]) -> BPE_Tokenizer:
    #import pdb; pdb.set_trace()
    #Load the input file
    PAT_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    text = ""
    with open(input_path, 'r') as file:
        text = file.read()

    #Pre-Tokenize the input file
    tokens = re.findall(PAT_regex, text)

    #Create the initial vocabulary
    vocab = {x: bytes([x]) for x in range(256)}
    for index, token in enumerate(special_tokens):
        utf8_token = token.encode("utf-8")
        vocab[index+256] = utf8_token
    
    num_merge_tokens = vocab_size - len(vocab)
    #num_merge_tokens = 3

    #Frequency of each token
    token_freq = {}
    for token in tokens:
        token = [ch.encode('utf-8') for ch in token]
        token = tuple(token)
        if token in token_freq:
            token_freq[token] += 1
        else:
            token_freq[token] = 1
    
    #Create the byte pairs
    byte_pairs_freq = {}
    byte_pairs_tokens = {}
    for token, freq in token_freq.items():
        add_byte_pairs(byte_pairs_freq, byte_pairs_tokens, token, freq)
    
    #Merge the byte pairs 
    merges = []
    for i in range(num_merge_tokens):
        merged_byte_pair, _ = max_freq_byte_pair(byte_pairs_freq)
        #Add the merged byte pair to the list of merges
        merges.append(merged_byte_pair)
        # if(merged_byte_pair == (b'h', b'e')):
        #     import pdb; pdb.set_trace()
        #add the merged byte pair to the vocabulary
        token_str = merged_byte_pair[0].decode('utf-8') + merged_byte_pair[1].decode('utf-8')
        encoded_token = token_str.encode('utf-8')
        vocab[len(vocab)] = encoded_token
        
        #Update the byte pairs by removing the contribution from tokens containing merged byte pair 
        new_token_freq_list = []
        byte_pairs_tokens_entry = byte_pairs_tokens[merged_byte_pair].copy()
        for token, freq in set(byte_pairs_tokens_entry):
            # if(token == (b' ', b'w', b'h', b'e', b't', b'h', b'e', b'r')):
            #     import pdb; pdb.set_trace()
            
            sub_byte_pairs(byte_pairs_freq, byte_pairs_tokens, token, freq)
            
                
            #Create a new token with merged byte pair
            token_list = list(token)
            len_token = len(token_list)
            new_token = []
            new_token_len = 0
            k = 0
            while k < (len_token - 1):
                if token_list[k] == merged_byte_pair[0] and token_list[k+1] == merged_byte_pair[1]:
                    token_str = merged_byte_pair[0].decode('utf-8') + merged_byte_pair[1].decode('utf-8')
                    new_token.append(token_str.encode('utf-8'))
                    k += 2
                    new_token_len += 2
                else:
                    new_token.append(token_list[k])
                    k +=1
                    new_token_len += 1
            if (k == len_token - 1):
                if(new_token_len != len_token):
                    new_token.append(token_list[k])
                # if not(token_list[k-1] == merged_byte_pair[0] and token_list[k] == merged_byte_pair[1]):
                    # new_token.append(token_list[k])

                    
            new_token = tuple(new_token)
            new_token_freq_list.append((new_token, freq))

        #Update the byte pairs by adding the contribution from new tokens containing merged byte pair
        new_token_str = ""
        orig_token_str = ""
        for m in range(len(new_token)):  
            new_token_str += new_token[m].decode('utf-8')
        for m in range(len(token)):
            orig_token_str += token[m].decode('utf-8')
        #print(f'New_token_str = {new_token_str}, Orig_token_str = {orig_token_str}')
        #print(f"New_token = {new_token}, Orig_token = {token}")
        try:
            assert new_token_str == orig_token_str
        except AssertionError:
            print(f'Assertion failed!')
            print(f'New_token_str = {new_token_str}, Orig_token_str = {orig_token_str}')
            print(f"New_token = {new_token}, Orig_token = {token}")

        for token, freq in new_token_freq_list:
            add_byte_pairs(byte_pairs_freq, byte_pairs_tokens, token, freq)
        
    return BPE_Tokenizer(vocab, merges)


        



def add_byte_pairs(byte_pairs_freq: Dict[bytes, int], byte_pairs_tokens: Dict[bytes, List[Tuple[bytes]]], token: Tuple[bytes], freq: int):
    token_shifted = token[1:]
    byte_pairs = zip(token, token_shifted)
    for byte_pair in byte_pairs:
        if byte_pair in byte_pairs_freq:
            byte_pairs_freq[byte_pair] += freq
        else:
            byte_pairs_freq[byte_pair] = freq

        if byte_pair in byte_pairs_tokens:
            byte_pairs_tokens[byte_pair].append((token, freq))
        else:
            byte_pairs_tokens[byte_pair] = [(token, freq)]
        #compare_freq(byte_pairs_freq, byte_pairs_tokens, byte_pair)

def sub_byte_pairs(byte_pairs_freq: Dict[bytes, int], byte_pairs_tokens: Dict[bytes, List[str]], token: str, freq: int):
    token_shifted = token[1:]
    byte_pairs = zip(token, token_shifted)
    for byte_pair in byte_pairs:
        if byte_pair in byte_pairs_freq:
            byte_pairs_freq[byte_pair] -= freq
            if byte_pairs_freq[byte_pair] == 0:
                byte_pairs_freq.pop(byte_pair)

        if byte_pair in byte_pairs_tokens:
            if (token, freq) in byte_pairs_tokens[byte_pair]:
                byte_pairs_tokens[byte_pair].remove((token, freq))
            if len(byte_pairs_tokens[byte_pair]) == 0:
                byte_pairs_tokens.pop(byte_pair)
        #compare_freq(byte_pairs_freq, byte_pairs_tokens, byte_pair)

def max_freq_byte_pair(byte_pairs_freq: Dict[bytes, int]) -> Tuple[bytes, int]:
    #max_key = max(byte_pairs_freq.items(), key=lambda x: (x[1],x[0]))[0]
    #return max_key, 1                                                      
    max_freq = 0
    max_freq_byte_pair = None
    for byte_pair, freq in byte_pairs_freq.items():
        if freq > max_freq or (freq == max_freq and byte_pair > max_freq_byte_pair):
            max_freq = freq
            max_freq_byte_pair = byte_pair
    return max_freq_byte_pair, max_freq

    
def compare_freq(byte_pairs_freq: Dict[bytes, int], byte_pairs_tokens: Dict[bytes, List[str]], byte_pair) -> bool:
    # for byte_pair, freq in byte_pairs_freq.items():
    a = 0
    b = 0
    if byte_pair in byte_pairs_tokens:
        a = sum([token_freq for token, token_freq in byte_pairs_tokens[byte_pair]])
    if byte_pair in byte_pairs_freq:
        b = byte_pairs_freq[byte_pair]
    #print(f'a = {a}, b = {b}')
    if a != b:
        print(f'a = {a}, b = {b}')
        print(f'Byte pair = {byte_pair}')
        #import pdb; pdb.set_trace()
        #return False
    #import pdb; pdb.set_trace()
    return True
        