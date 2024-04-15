import torch
import torch.nn as nn
import numpy as np
import argparse
import os

from BPE_Tokenizer import bpe_train, BPE_Tokenizer, Tokenizer
from Others import data_loading, load_checkpoint, save_checkpoint, cross_entropy
from Transformer import transformer_lm
from Adam import adam

class trainer():
    def __init__(self, args):
        self.vocab_path = args.vocab_path
        self.merges_path = args.merges_path
        self.special_tokens = args.special_tokens

        self.tokenizer = Tokenizer.from_files(self.vocab_path, self.merges_path, self.special_tokens)
        self.input_train_file = args.input_train_file
        self.tokenized_input_train_file = args.tokenized_input_train_file + '.npy'
        self.input_val_file = args.input_val_file
        self.tokenized_input_val_file = args.tokenized_input_val_file + '.npy'
        self.vocab_size = args.vocab_size
        self.context_length = args.context_length
        self.d_model = args.d_model
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.d_ff = args.d_ff
        self.attn_pdrop = args.attn_pdrop
        self.residual_pdrop = args.residual_pdrop
        self.device = args.device


    def generate_tokenized_data(self):
        with open(self.input_train_file, 'r') as f:
            text = f.read()
        ids = self.tokenizer.encode(text)
        ids_np = np.array(ids, dtype=np.int32)
        np.save(self.tokenized_input_train_file, ids_np)

        with open(self.input_val_file, 'r') as f:
            text = f.read()
        ids = self.tokenizer.encode(text)
        ids_np = np.array(ids, dtype=np.int32)
        np.save(self.tokenized_input_val_file, ids_np)

    def load_tokenized_data(self, train=True):
        if train:
            self.tokenized_input_file = self.tokenized_input_train_file
        else:
            self.tokenized_input_file = self.tokenized_input_val_file
        temp_array = np.load(self.tokenized_input_file, mmap_mode='r')  # This loads the header to determine shape and dtype
        dtype = temp_array.dtype
        shape = temp_array.shape

        # Now use np.memmap to map the array stored in the .npy file directly
        mapped_array = np.memmap(self.tokenized_input_file, dtype=dtype, mode='r', shape=shape)
        return mapped_array
    
    def get_model(self):
        model = transformer_lm(self.vocab_size, self.context_length, 
                               self.num_layers, self.d_model, self.num_heads, 
                               self.d_ff, self.attn_pdrop, self.residual_pdrop)
        model.to(self.device)
        return model
    
    def train_model(self, args):
        dir_name = args.experiment_name
        os.makedirs(dir_name)
        tokenized_train_data = self.load_tokenized_data(train=True)
        tokenized_val_data = self.load_tokenized_data(train=False)
        model = self.get_model()
        optimizer = adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay, betas = tuple(args.betas), eps = args.eps)
        iteration = 0
        if(args.load_checkpoint_path != ''):
            iteration = load_checkpoint(args.load_checkpoint_path, model, optimizer)
        for i in range(iteration, args.iters):
            optimizer.zero_grad()
            input, targets = data_loading(tokenized_train_data, args.batch_size, args.context_length, args.device)
            predictions = model(input)
            loss = cross_entropy(predictions, targets)
            print(f'Iteration: {i}, Training Loss: {loss.item()}')
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                if i % args.validation_interval == 0:
                    input_val, targets_val = data_loading(tokenized_val_data, args.batch_size, args.context_length, args.device)
                    predictions_val = model(input_val)
                    val_loss = cross_entropy(predictions_val, targets_val)
                    print(f'Iteration: {i}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}')
                    save_checkpoint(model, optimizer, i, f'./{dir_name}/checkpoint_{i}.pt')
        
        with torch.no_grad():
            input_val, targets_val = data_loading(tokenized_val_data, args.batch_size, args.context_length, args.device)
            predictions_val = model(input_val)
            val_loss = cross_entropy(predictions_val, targets_val)
            print(f'Iteration: {args.iters}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}')
            save_checkpoint(model, optimizer, args.iters, f'./{dir_name}/checkpoint_{args.iters}.pt')
        

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_path", type=str, default='./test_akshit/TinyStoriesV2-GPT4-train.txt_my_vocab.json', help="Path to the vocab file")
    parser.add_argument("--merges_path", type=str, default='./test_akshit/TinyStoriesV2-GPT4-train.txt_my_merges.txt', help="Path to the merges file")
    parser.add_argument("--special_tokens", nargs='+', default=["<|endoftext|>"], help="Special tokens to be used in the tokenizer")
    parser.add_argument("--input_train_file", type=str, default='/data/TinyStoriesV2-GPT4-train.txt', help="Path to the input file")
    parser.add_argument("--input_val_file", type=str, default='/data/TinyStoriesV2-GPT4-valid.txt', help="Path to the input file")
    parser.add_argument("--tokenized_input_train_file", type=str, default='./cs336_basics/mydata/TinyStoriesV2-GPT4-train_tokenized', help="Path to the tokenized file")
    parser.add_argument("--tokenized_input_val_file", type=str, default='./cs336_basics/mydata/TinyStoriesV2-GPT4-val_tokenized', help="Path to the tokenized file")
    parser.add_argument("--tokenize", type=bool, default=False, help="Whether to tokenize the input file")
    parser.add_argument("--train", type=bool, default=False, help="Whether to train the tokenizer")
    parser.add_argument("--context_length", type=int, default=512, help="Context length for the language model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training the language model")
    parser.add_argument("--device", type=str, default='cuda', help="Device to be used for training the model")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size for the transformer")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension for the transformer")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers in the transformer")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads in the transformer")
    parser.add_argument("--d_ff", type=int, default=512, help="Feed forward dimension in the transformer")
    parser.add_argument("--attn_pdrop", type=float, default=0.0, help="Dropout probability for attention")
    parser.add_argument("--residual_pdrop", type=float, default=0.0, help="Dropout probability for residual connections")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer")
    parser.add_argument("--betas", nargs='+', default=[0.9, 0.999], help="Betas for the optimizer")
    parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon for the optimizer")
    parser.add_argument("--iters", type=int, default=10, help="Number of epochs for training the model")
    parser.add_argument("--load_checkpoint_path", type=str, default='', help="Path to load checkpoint")
    parser.add_argument("--validation_interval", type=int, default=3, help="number of iterations before validation loss is calculated for the model")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment")

    return parser.parse_args()

def main():
    args = get_args()
    trainer_obj = trainer(args)
    if args.tokenize:
        trainer_obj.generate_tokenized_data()
    if args.train:
        trainer_obj.train_model(args)
        

if __name__ == '__main__':
    main()
