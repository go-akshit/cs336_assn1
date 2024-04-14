import regex as re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Iterator
import sys
sys.path.insert(0, '../cs336_assn1/tests')
from common import gpt2_bytes_to_unicode

@dataclass
class BPE_Tokenizer:
    vocab: Dict[int, bytes]
    merges: List[Tuple[bytes, bytes]]

def bpe_train(input_path: str, vocab_size: int, special_tokens: list[str]) -> BPE_Tokenizer:
    #import pdb; pdb.set_trace()
    #Load the input file
    PAT_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    text = ""
    tokens = []
    with open(input_path, 'r') as file:
        text = file.readline()
        while text:
            # Get rid of special tokens as I add them later anyway
            for token in special_tokens:
                text = text.replace(token, "")

            #Pre-Tokenize the input file
            tokens.extend(re.findall(PAT_regex, text))
            text = file.readline()

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
        for token, freq in byte_pairs_tokens_entry:
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

    resulting_tokenizer = BPE_Tokenizer(vocab, merges)
    #import pdb; pdb.set_trace()
    vocab_file_name = str(input_path).split("/")[-1] + "_my_vocab.json" 
    merge_file_name = str(input_path).split("/")[-1] + "_my_merges.txt" 
    save_bpe_tokenizer(resulting_tokenizer, vocab_file_name, merge_file_name)
    return resulting_tokenizer

def save_bpe_tokenizer(bpe_tokenizer: BPE_Tokenizer, vocab_output_path: str, merges_output_path: str):
    dict = gpt2_bytes_to_unicode()
    vocab = bpe_tokenizer.vocab
    merges = bpe_tokenizer.merges
    with open(vocab_output_path , 'w') as file:
        for key, value in vocab.items():
            value_str = ''.join(dict[byte] for byte in value)
            file.write(f'{key} {value_str}\n')
    with open(merges_output_path, 'w') as file:
        for each in merges:
            file.write(f'{each}\n')

        



def add_byte_pairs(byte_pairs_freq: Dict[bytes, int], byte_pairs_tokens: Dict[bytes, List[Tuple[bytes]]], token: Tuple[bytes], freq: int):
    token_shifted = token[1:]
    byte_pairs = zip(token, token_shifted)
    for byte_pair in byte_pairs:
        if byte_pair in byte_pairs_freq:
            byte_pairs_freq[byte_pair] += freq
        else:
            byte_pairs_freq[byte_pair] = freq

        if byte_pair in byte_pairs_tokens:
            if (token, freq) not in byte_pairs_tokens[byte_pair]:
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


class Tokenizer: 
    def __init__(self, vocab, merges, special_tokens = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        if special_tokens is not None:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
    
    #def from_files(cls, vocab_filepath, merges_filepath, special_tokens = None):

    def encode_main(self, text: str) -> List[int]:
        encoded_token_ints = []

        #Pre-Tokenize the input text
        if self.special_tokens is not None:
            if text in self.special_tokens:
                tokens = [text]
                encoded = []
                for each in tokens:
                    byte_encoding = each.encode('utf-8')
                    for key, value in self.vocab.items():
                        if byte_encoding == value:
                            encoded.append(key)
                assert len(encoded) == 1
                encoded_token_ints.extend(encoded)
                return encoded_token_ints

        PAT_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        tokens = re.findall(PAT_regex, text)
        

        for item in tokens:
            token = []
            for ch in item:
                byte_encoding = ch.encode('utf-8')
                for byte in byte_encoding:
                    token.append(bytes([byte]))
      
            #For each byte pair in the token, find the byte pair with the smallest index in the merges list for merging
            while True:
                min_pos = len(self.merges)
                merged_byte_pair = None
                token_shifted = token[1:]
                for byte_pair in zip(token, token_shifted):
                    try:
                        pos = self.merges.index(byte_pair)
                    except: 
                        pos = -1

                    if pos >= 0:
                        if pos <= min_pos:
                            min_pos = pos
                            merged_byte_pair = byte_pair

                if merged_byte_pair is None:
                    break
                else:
                    #Create a new token with merged byte pair
                    token_list = list(token)
                    len_token = len(token_list)
                    new_token = []
                    new_token_len = 0
                    k = 0
                    while k < (len_token - 1):
                        if token_list[k] == merged_byte_pair[0] and token_list[k+1] == merged_byte_pair[1]:
                            # token_str = merged_byte_pair[0].decode('utf-8') + merged_byte_pair[1].decode('utf-8')
                            # new_token.append(token_str.encode('utf-8'))
                            temp = []
                            temp.extend(list(merged_byte_pair[0]))
                            temp.extend(list(merged_byte_pair[1]))
                            # merged = bytes([list(merged_byte_pair[0])[0], list(merged_byte_pair[1])[0]])
                            merged = bytes(temp)
                            new_token.append(merged)
                            k += 2
                            new_token_len += 2
                        else:
                            new_token.append(token_list[k])
                            k +=1
                            new_token_len += 1
                    if (k == len_token - 1):
                        if(new_token_len != len_token):
                            new_token.append(token_list[k])
                    token = new_token
            
            #Encode the token
            encoded = []
            for each in token:
                for key, value in self.vocab.items():
                    if each == value:
                        encoded.append(key)
            assert len(encoded) == len(token)
            encoded_token_ints.extend(encoded)
        
        return encoded_token_ints
    
    def encode(self, text: str) -> List[int]:
        split_regex = [text]
        
        if self.special_tokens is not None:
            if len(self.special_tokens) != 0:
                escaped_special_tokens = [re.escape(token) for token in self.special_tokens]
                pattern = '|'.join(escaped_special_tokens)
                pattern += '|\n'
                split_regex = re.findall(pattern + '|.+?(?=' + pattern + '|$)', text)
            else:
                split_regex = [text]
        encoded_token_ints = []
        for item in split_regex:
            encoded_token_ints.extend(self.encode_main(item))
        return encoded_token_ints

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        byte_string = b"".join([self.vocab.get(id, b'\xef\xbf\xbd') for id in ids])
        text = byte_string.decode('utf-8', errors='replace')
        return text