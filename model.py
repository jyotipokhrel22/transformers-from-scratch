import torch # main pytorch library
import torch.nn as nn # pytorch's neural network module
import math

class InputEmbeddings(nn.Module): # defines a custom pytorch module i.e. a model component

    # Constructor, runs when an object is created
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__() # initializes the parent nn.Module

        # d_model: how many numbers each token is represented by = 512
        # vocab_size: how many distinct tokens you can look up

       # self means this current object
        self.d_model = d_model # stores model's dimension inside this object
        self.vocab_size = vocab_size # words in the vocabulary
        
        # creates an actual embedding layer, i/p integer token ID, o/p dense vector
        self.embedding = nn.Embedding(vocab_size, d_model) 

    
    # takes token as an input, x = Token IDs
    def forward(self, x):
        return self.embedding(x) * math.sqrt(d_model)

class PositionalEncodings(nn.Module):

    def __init__(self, d_model: int, seq_length: int, dropout: float):
        super().__init__()

        self.d_model = d_model
        self.seq_length = seq_length # maximum sequence length
        self.dropout = nn.Dropout(dropout) # create a dropout layer, to prevent overfitting, e.g. dropout = 0.1, means each value has 90% chance of being kept, other = 0

        # Create a matrix of shape (seq_length, d_model)
        pe = torch.zeros(seq_length, d_model)

        # Create a vector of shape(seq_length, 1)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1) # unsqueeze(1) -> turns row vector to column vector

        # creating the scaling values used in positional encoding, sin(position * something)
        # torch.arange(0, d_model, 2) because, even indices for sin, odd indices for cos

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)# 0, 2, 4 ...
        pe[:, 1::2] = torch.cos(position * div_term) # 1, 3, 5 ..

        # add batch dimension to tensor 
        pe = pe.unsqueeze(0) # Added extra dimension, 1, seq_length, d_model

        self.register_buffer('pe', pe) # stores positional encodings in the model

    def forward(self, x): # add  positional encodings to embeddings

        # slice pe table to match current sequence length
        # x + .. adds positional encodings to the token embeddings
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # not computing gradients, i.e. not changing values
        return self.dropout(x)

class LayerNormalization(nn.Module): # normalizes each token vector using mean and standard deviation

    def __init__(self, eps: float = 10**-6): # numerically stable value
        super().__init__()

        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # this makes the parameters learnable,multiplied, scale parameter
        self.bias = nn.Parameter(torch.zeros(1)) # additive, shift parameter

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True) # computes mean across the last dimension, each token vector gets its own mean
        std = x.std( dim = -1, keepdim=True) # computes the std across that same last dimension
        return self.alpha * (x - mean) / (std + self.eps) + self.bias # formula



# attention mixes info across tokens because each token rebuilds its representation by taking a weighted sum of other tokens value vectors.
# attention mixes information across tokens, feed-forward transforms each token vector more deeply
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float): # d_ff = hidden size of feed forward network = 2048
        super().__init__()

        # fully connected layer, vector size from d_model -> d_ff
        # [batch_size, seq_length, d_model] -> [batch_size, seq_length, d_ff]
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 & B1 , bias is by default true in nn.Linear
        self.dropout = nn.Dropout(dropout) # some values randomly zeroed during training
        # d_ff -> d_model
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 & B2

    def forward(self, x):
        # Batch, seq_length, d_model -> linear_1 applied = batch, seq_length, d_ff -> linear_2 applied = batch, seq_length, d_model

        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        # relu -ve = 0, + ve as it is stays

