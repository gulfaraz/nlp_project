import torch
import torch.nn as nn
from torch.autograd import Variable

class GRU_Regression(nn.Module):
    #output_dim should be one according to the current dimention
    def __init__(self, vocab_size, embedding_dim, img_feature_size, output_dim, hidden_size, batch_size):
        super(GRU_Regression, self).__init__()

        self.output_vector_size = output_dim
        self.vocab_size =  vocab_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden = self.init_hidden()
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first = True)
        self.linear = nn.Linear(hidden_size, img_feature_size)

    def forward(self, text_input, last_words):
        embeds = self.embeddings(text_input) 
        
        gru_out, self.hidden = self.gru(embeds, self.hidden)
        last_out = torch.gather(gru_out, 1, last_words.view(-1,1,1).expand(self.batch_size, 1, self.hidden_size) - 1)

        output = self.linear(last_out).squeeze()

        return output

    def init_hidden(self):
        return Variable(torch.zeros(self.batch_size, 1, self.hidden_size))
