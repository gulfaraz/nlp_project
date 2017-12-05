import torch
import torch.nn as nn

class Regression(nn.Module):
    #output_dim should be one according to the current dimention
    def __init__(self, vocab_size, embedding_dim, img_feature_size, output_dim):
        super(Regression, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, img_feature_size)

        #self.linear2 = nn.Linear(128, vocab_size)
        #self.activation_function = nn.Softmax()
        self.output_vector_size = output_dim
        self.vocab_size =  vocab_size

        #Inputs should be in Tensor form
        #image_input is list of tensors
    def forward(self, text_input):
        embeds = self.embeddings(text_input) / self.vocab_size
        sum_embeds = torch.sum(embeds, 1)
        output = self.linear1(sum_embeds)

        return output
