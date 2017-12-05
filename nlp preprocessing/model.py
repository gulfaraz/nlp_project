import torch
import torch.nn as nn

class NaiveCBOW(nn.Module):
    #output_dim should be one according to the current dimention
    def __init__(self, vocab_size, embedding_dim, img_feature_size, output_dim):
        super(NaiveCBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        #TODO check if nn.Parameter or nn.Linear
        self.linear = nn.Linear((embedding_dim + img_feature_size), 1)
        self.activation_function = nn.Softmax()
        self.output_vector_size = output_dim
        self.vocab_size =  vocab_size
        self.img_feature_size = img_feature_size

        #Inputs should be in Tensor form
        #image_input is list of tensors
    def forward(self, text_input, image_input):
        embeds = self.embeddings(text_input)
        sum_embeds = torch.sum(embeds, 1) / self.vocab_size 
        sum_embeds = sum_embeds.unsqueeze(-1).transpose(2,1)

        sum_embeds_block = sum_embeds.repeat(1, self.output_vector_size, 1)
        image_block = image_input
        block = torch.cat((sum_embeds_block, image_block), 2)

        score = self.linear(block).squeeze()
        output = self.activation_function(score)
        
        return output
