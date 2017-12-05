import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class Evaluate():
    def __init__(self, model, img_features, minibatch, preprocess, image_feature_size):
        self.model = model
        self.img_features = img_features
        self.minibatch = minibatch
        self.preprocess = preprocess
        self.criterion = nn.MSELoss()
        self.image_feature_size = image_feature_size

    def __call__(self, data):
        TOP1 = 0.0
        TOP5 = 0.0

        for batch in self.minibatch(data):
            #Outputs matrices of batch size
            text_features, h5_ids, correct_index = self.preprocess(batch)

            lookup_text_tensor = Variable(torch.LongTensor([text_features])).squeeze()
            scores = self.model(lookup_text_tensor)

            #create list of: batch size * 10 * 2048
            for obs, img_ids in enumerate (h5_ids):

                lookup_img_list = [Variable(torch.from_numpy(self.img_features[h5_id])).view(1,-1) for h5_id in img_ids]
                losses = np.empty([10,1])

                for index, single_image in enumerate(lookup_img_list):
                    #Calculate loss per image
                    loss = self.criterion(scores[obs], single_image)
                    losses[index] = loss.data.numpy()[0]

                #Get TOP1 and TOP5
                if np.argmin(losses) == correct_index[obs]:
                    TOP1 += 1

                if correct_index[obs] in np.argsort(losses, axis = 0)[:5, ]:
                    TOP5 += 1

        return TOP1, TOP5, TOP1/len(data), TOP5/len(data), len(data)

    def calculate_loss(self, data):

        c_loss = 0.0
        for batch in self.minibatch(data):

            text_features, h5_ids, correct_index = self.preprocess(batch)
            lookup_text_tensor = Variable(torch.LongTensor([text_features])).squeeze()

            target = np.empty([len(batch), self.image_feature_size])

            for obs, img_ids in enumerate (h5_ids):
                target[obs] = self.img_features[img_ids[correct_index[obs]]]

            target = Variable(torch.from_numpy(target).type(torch.FloatTensor))
            prediction = self.model(lookup_text_tensor)
            loss = self.criterion(prediction, target)
            c_loss += loss.data[0] 

        return c_loss / len(data)
