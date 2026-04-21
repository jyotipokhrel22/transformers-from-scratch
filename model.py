import torch # main pytorch library
import torch.nn as nn # pytorch's neural network module

class InputEmbeddings(nn.Module): # defines a custom pytorch module i.e. a model component

    # Constructor, runs when an object is created
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__() # initializes the parent nn.Module

        # d_model: how many numbers each token is represented by
        # vocab_size: how many distinct tokens you can look up

       # self means this current object
        self.d_model = d_model # stores model's dimension inside this object
        self.vocab_size = vocab_size # words in the vocabulary
        
        # creates an actual embedding layer, i/p integer token ID, o/p dense vector
        self.embedding = nn.Embedding(vocab_size, d_model) 

    def forward(self, x):
        return self.embedding(x) * math.sqrt(d_model)