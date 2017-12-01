import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class NaiveCBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, img_feature_size, output_dim):
        super(NaiveCBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear((embedding_dim + img_feature_size), 1)
        self.activation_function = nn.Softmax()
        self.output_vector_size = output_dim

    def forward(self, text_input, image_input):
        embeds = self.embeddings(text_input)
        sum_embeds = torch.sum(embeds, 1)
        sum_embeds_block = sum_embeds.repeat(self.output_vector_size, 1)
        image_block = torch.stack(image_input).squeeze()
        block = torch.cat((sum_embeds_block, image_block), 1)
        score = self.linear(block)
        output = self.activation_function(score.view(1, -1))
        return output

    def evaluate(self, model, data, img_features):
        """Evaluate a model on a data set."""
        TOP1 = 0.0
        TOP5 = 0.0

        for text_features, h5_ids, correct_index in data:
            #turn text data into tensors
            lookup_text_tensor = Variable(torch.LongTensor([text_features]))
            #Turn image features into Tensors
            lookup_img_list = [Variable(torch.from_numpy(img_features[h5_id])).view(1,-1) for h5_id in h5_ids]
            #Pass into the model
            scores = model(lookup_text_tensor, lookup_img_list)
            predict = scores.data.numpy().argmax(axis=1)[0]

            if predict == correct_index:
                TOP1 += 1

            if correct_index in np.argsort(-scores.data.numpy())[0, :5]:
                TOP5 += 1

        return TOP1, TOP5, TOP1/len(data), TOP5/len(data), len(data)
