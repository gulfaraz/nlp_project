import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class Regression(nn.Module):
    def __init__(self, vocab_size, embedding_dim, img_feature_size, output_dim):
        super(Regression, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, img_feature_size)
        self.output_vector_size = output_dim
        self.vocab_size =  vocab_size

    def forward(self, text_input):
        embeds = self.embeddings(text_input)
        sum_embeds = torch.sum(embeds, 1) / self.vocab_size 
        output = self.linear1(sum_embeds)
        return output

    def evaluate(self, model, data, img_features):
        """Evaluate a model on a data set. From the range of images pick the one with the smallest vector distance"""
        TOP1 = 0.0
        TOP5 = 0.0
        criterion = nn.MSELoss()
        #criterion = nn.L1Loss()

        for text_features, h5_ids, correct_index in data:
            #turn text data into tensors
            lookup_text_tensor = Variable(torch.LongTensor([text_features]))

            #Turn image features into Tensors
            lookup_img_list = [Variable(torch.from_numpy(img_features[h5_id])).view(1,-1) for h5_id in h5_ids]
            target = lookup_img_list[correct_index]

            #Pass into the model
            scores = model(lookup_text_tensor)
            losses = np.empty([10,1])
            #evaluate
            for index in range(len(lookup_img_list)):
                loss = criterion(scores, lookup_img_list[index])
                losses[index] = loss.data.numpy()[0]

            if np.argmin(losses) == correct_index:
                TOP1 += 1

            if correct_index in np.argsort(losses, axis = 0)[:5, ]:
                TOP5 += 1

        return TOP1, TOP5, TOP1/len(data), TOP5/len(data), len(data)
