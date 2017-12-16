import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import Counter

class Evaluate():
    def __init__(self, model, img_features, minibatch, preprocess, image_feature_size, i2w):
        self.model = model
        self.img_features = img_features
        self.minibatch = minibatch
        self.preprocess = preprocess
        self.criterion = nn.MSELoss()
        self.image_feature_size = image_feature_size
        self.i2w = i2w


    def __call__(self, data, batch_size):
        TOP1 = 0.0
        TOP5 = 0.0
        criterion = nn.MSELoss()


        for batch in self.minibatch(data, batch_size = batch_size):

            self.model.hidden = self.model.init_hidden()

            #Outputs matrices of batch size
            text_features, h5_ids, correct_index, last_words = self.preprocess(batch)

            lookup_text_tensor = Variable(torch.LongTensor([text_features])).squeeze()
            last_words = Variable(torch.LongTensor(last_words))

            scores = self.model(lookup_text_tensor, last_words)

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

    def calculate_loss(self, data, batch_size):
        c_loss = 0.0
        for batch in self.minibatch(data, batch_size = batch_size):

            self.model.hidden = self.model.init_hidden()

            text_features, h5_ids, correct_index, last_words = self.preprocess(batch)
            lookup_text_tensor = Variable(torch.LongTensor([text_features])).squeeze()

            target = np.empty([len(batch), self.image_feature_size])
            last_words = Variable(torch.LongTensor(last_words))

            for obs, img_ids in enumerate (h5_ids):
                target[obs] = self.img_features[img_ids[correct_index[obs]]]

            target = Variable(torch.from_numpy(target).type(torch.FloatTensor))
            prediction = self.model(lookup_text_tensor, last_words)
            loss = self.criterion(prediction, target)
            c_loss += loss.data[0] 

        return c_loss / len(data)


    def create_dic(self, data, batch_size):

        wrong_word_counter_TOP1 = Counter()
        correct_word_counter_TOP1 = Counter()

        wrong_word_counter_TOP5 = Counter()
        correct_word_counter_TOP5 = Counter()

        score_confidence = []
        correct = []


        for batch in self.minibatch(data, batch_size = batch_size):

            self.model.hidden = self.model.init_hidden()

            #Outputs matrices of batch size
            text_features, h5_ids, correct_index, last_words = self.preprocess(batch)

            lookup_text_tensor = Variable(torch.LongTensor([text_features])).squeeze()
            last_words = Variable(torch.LongTensor(last_words))

            scores = self.model(lookup_text_tensor, last_words)

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

                    correct.append(True)

                    sum_losses = np.sum(losses)
                    losses_percentage = losses / sum_losses
                    nr1 = losses_percentage[np.argmin(losses)]
                    nr2 = losses_percentage[np.argsort(losses)[1]]
                    
                    score_confidence.append(- (nr1 - nr2))

                    #Include correct words in Counter
                    for word in text_features[obs]:
                        correct_word_counter_TOP1[self.i2w[word]] += 1
                else:
                    correct.append(False)

                    for word in text_features[obs]:
                        wrong_word_counter_TOP1[self.i2w[word]] += 1
                    

                if correct_index[obs] in np.argsort(losses, axis = 0)[:5, ]:
              #Include correct words in Counter
                    for word in text_features[obs]:
                        correct_word_counter_TOP5[self.i2w[word]] += 1
                else: 
                    for word in text_features[obs]:
                        wrong_word_counter_TOP5[self.i2w[word]] += 1   
                    

        return (correct_word_counter_TOP1, wrong_word_counter_TOP1, correct_word_counter_TOP5, wrong_word_counter_TOP5, score_confidence, correct)
