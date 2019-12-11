import os
import math
import random
import spacy
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import numpy as np
import nltk
from io import open
import unicodedata
import string
import re
import random
import os


# # CONSTANTS

# In[15]:


PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

PAD, SOS, EOS, UNK = "<PAD>", "<SOS>", "<EOS>", "<UNK>"

# define batch size
BATCH_SIZE = 32
print('Using batch size:', BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# # VOCAB

# In[4]:

class Vocab:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0
        
        self.add_word(PAD)
        self.add_word(SOS)
        self.add_word(EOS)
        self.add_word(UNK)
        

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# # DATA PREP

# In[23]:


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize(s):
    s = re.sub("ß", "ss", s)
    s = re.sub("ä", "ae", s)
    s = re.sub("ö", "oe", s)
    s = re.sub("ü", "ue", s)
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s.split(' ')

def sub_unks(token_array, vocab):
    words = set(vocab.index2word.values())
    return [t if t in words else UNK for t in token_array ]

def insert_tags(token_array):
    return [SOS] + token_array + [EOS]

def tokenize_string(vocab, s):
    # ex s = 'and the evaluation'
    tokens = normalize(s)
    tokens = insert_tags(tokens)
    tokens = sub_unks(tokens, vocab)
    return tokens


# # MODEL

# In[12]:


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)
    
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] =                 math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] =                 math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len],         requires_grad=False).to(device)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    
class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True))         / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model).to(device)
    
    def forward(self, x, e_outputs, src_mask, trg_mask):
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
            src_mask))
            x2 = self.norm_3(x)
            x = x + self.dropout_3(self.ff(x2))
            return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)
    
# build an encoder layer with one multi-head attention layer and one # feed-forward layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads)
        self.decoder = Decoder(trg_vocab, d_model, N, heads)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output
    
import numpy as np
import copy 

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output


def create_masks(input_seq, target_seq):
    input_pad = PAD_token
    # creates mask with 0s wherever there is padding in the input
    input_msk = (input_seq != input_pad).unsqueeze(1)
    
    # create mask as before
    target_pad = PAD_token
    target_msk = (target_seq != target_pad).byte().unsqueeze(1)
    size = target_seq.size(1) # get seq_len for matrix
    nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0).byte().to(device)
    target_msk = target_msk & nopeak_mask
    
    return input_msk, target_msk

# We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# # PREDICTION

# In[31]:


def remove_duplicates(vocab, id_array):
    # remove tokens which are repeated in succession
    id_array = [id_array[i] for i in range(len(id_array)) if i == 0 or id_array[i-1] != id_array[i]]
    return id_array

def beautify_token_array_to_str(token_array):
    import re
    # Convert to string, strip training spaces
    string = ' '.join(token_array).strip()
    '''Replaces common contractions such as don't, ain't, isn't etc.'''
    string = re.sub('ai n t', "ain't", string)
    string = re.sub('do n t', "don't", string)
    string = re.sub('is n t', "isn't", string)
    string = re.sub('i m', "i'm", string)
    string = re.sub('was n t', "wasn't", string)
    string = re.sub('wo n t', "won't", string)
    string = re.sub('should n t', "shouldn't", string)
    return string

def tokens_to_tensor(LANG, tokens, reverse=False):
    idx = [LANG.word2index[t] for t in tokens]
    if reverse: idx = list(reversed(idx))
    return idx

def remove_tags(LANG, idx):
    ignore = [LANG.word2index[PAD], LANG.word2index[SOS], LANG.word2index[EOS]]
    return [i for i in idx if i not in ignore]

def tensor_to_tokens(LANG, tensor, reverse=False, remove_dups=True):
    tensor = remove_tags(LANG, tensor)
    if remove_dups: 
        tensor = remove_duplicates(LANG, tensor)
    tokens = [LANG.index2word[x] for x in tensor]
    tokens = list(reversed(tokens)) if reverse else tokens
    return tokens

                         
def get_pred_tar_pairs(batch_iter):
    'predict on batches and finds the untouched txt sentence which matches the target.'
    TARS = []
    PREDS = []
    SRCS = []

    with torch.no_grad():
        for batch in valid_iterator:
            # Prepare batch
            src, trg = batch
            src = src.transpose(0,1)
            trg = trg.transpose(0,1)
            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input)

            # Forward pass
            output = model(src, trg_input, src_mask, trg_mask)
            
            # Argmax over vocab and get the predicted tokens
            pred = output.argmax(dim=2)
            

            SRCS += [tensor_to_tokens(vocab, x.tolist()) for x in src]
            TARS += [tensor_to_tokens(vocab, x.tolist()) for x in trg]
            PREDS += [tensor_to_tokens(vocab, x.tolist()) for x in pred]
    
    return SRCS, TARS, PREDS

from torchnlp.metrics import get_moses_multi_bleu
def bleu(tar, pred): 
    'Calculates moses bleu given two arrays of str tokens'
    tar, pred = ' '.join(tar), ' '.join(pred)
    return get_moses_multi_bleu([tar], [pred])

def paraphrase(model, vocab, src, max_len=80):
    model.eval()
    src = tokenize_string(vocab, src)
    src = Variable(torch.LongTensor([[vocab.word2index[tok] for tok in src]])).to(device)
    
    # Create mask and encode src
    src_mask = (src != PAD_token).unsqueeze(-2)
    encoder_outputs = model.encoder(src, src_mask)
    
    # Create decoder output as SOS tokens
    outputs = torch.zeros(max_len).type_as(src.data)
    outputs[0] = torch.LongTensor([vocab.word2index[SOS]])
        
    # Predict each token untill EOS
    for i in range(1, max_len):    
        trg_mask = np.triu(np.ones((1, i, i)), k=1).astype('uint8')
        trg_mask = Variable(torch.from_numpy(trg_mask) == 0).to(device)
        
        # Predict 
        out = model.out(model.decoder(outputs[:i].unsqueeze(0), encoder_outputs, src_mask, trg_mask))
        out = F.softmax(out, dim=-1)

        # Argmax over outputs
        outputs[i] = out[:, -1].argmax().item()
        
        # Check if EOS
        if outputs[i] == vocab.word2index[EOS]: 
            break

    # Convert to tokens, and lastly to beautiful string
    sentence = tensor_to_tokens(vocab, outputs[:i].tolist())
    return beautify_token_array_to_str(sentence)


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        print('Loaded device:', device)
        self.path = 'model_data/'
        self.vocab_path = self.path + 'vocab-200k.pickle'
        self.model_path = self.path + 'transformer-model-paraphrase.pt'
        self.vocab, self.model = self.load_model()

    def load_model(self):
        vocab = pickle.load(open(self.vocab_path, 'rb'))
        
        # Set up model
        d_model = 512
        heads = 8
        N = 6
        src_vocab = vocab.n_words
        trg_vocab = vocab.n_words
        model = Transformer(src_vocab, trg_vocab, d_model, N, heads).to(device)

         # Load Model params
        model.load_state_dict(torch.load(self.model_path, map_location=device))

        # Turn on eval mode (no randomness)
        model.eval()
        return vocab, model

    def paraphrase_sentence(self, input_str):
        pred = paraphrase(self.model, self.vocab, input_str)
        return pred