import torch
import torch.nn as nn
from torch.autograd import Variable

class GRU(nn.Module):
    #output_dim should be one according to the current dimention
    def __init__(self, vocab_size, embedding_dim, img_feature_size, output_dim, hidden_size, batch_size):
        super(GRU, self).__init__()
        #Parameters
        self.hidden_size = hidden_size
        self.output_vector_size = output_dim
        self.vocab_size = vocab_size
        self.batch_size = batch_size

        #Propagation layers
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first = True)
        self.hidden = self.init_hidden()
        self.linear = nn.Linear((hidden_size + img_feature_size), 1)
        self.activation_function = nn.Softmax()

        #Inputs should be in Tensor form
        #image_input is list of tensors
    def forward(self, text_input, image_input, last_words):
        embeds = self.embeddings(text_input)

        #Pass through GRU
        gru_out, self.hidden = self.gru(embeds, self.hidden)
        last_out = torch.gather(gru_out, 1, last_words.view(-1,1,1).expand(self.batch_size, 1, self.hidden_size) - 1)

        #Concatenate with pictures
        last_out_block = last_out.repeat(1, self.output_vector_size, 1)
        block = torch.cat((last_out_block, image_input), 2)

        score = self.linear(block).squeeze()
        score = self.activation_function(score)

        return score
        
    def init_hidden(self):
        return Variable(torch.zeros(self.batch_size, 1, self.hidden_size))
