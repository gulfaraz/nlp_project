import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class Evaluate():
    def __init__(self, model, img_features, minibatch, preprocess, image_feature_size, output_vector_size):
        self.model = model
        self.img_features = img_features
        self.minibatch = minibatch
        self.preprocess = preprocess
        self.criterion = nn.CrossEntropyLoss()
        self.image_feature_size = image_feature_size
        self.output_vector_size = output_vector_size

    def __call__(self, data, batch_size):
        """Evaluate a model on a batched data set."""
        TOP1 = 0.0
        TOP5 = 0.0

        for batch in self.minibatch(data, batch_size = batch_size):

            #Re-Initiate hidden state
            self.model.hidden = self.model.init_hidden()

            #load data with padding
            text_features, h5_ids, correct_index, last_words = self.preprocess(batch)

            #Prepare data for model input
            lookup_text_tensor = Variable(torch.LongTensor([text_features])).squeeze()
            

            full_img_batch = np.empty([len(batch), self.output_vector_size, self.image_feature_size])
            for obs, img_ids in enumerate(h5_ids):
                for index, h5_id in enumerate(img_ids):
                    full_img_batch[obs, index] = self.img_features[h5_id]

            full_img_batch = Variable(torch.from_numpy(full_img_batch).type(torch.FloatTensor))
            last_words = Variable(torch.LongTensor(last_words))

            #Run model for batch
            prediction = self.model(lookup_text_tensor, full_img_batch, last_words)
            prediction = prediction.data.numpy()
            
            #Calculate scores
            for obs, _ in enumerate(prediction):
                predict = prediction[obs]

                if np.argmax(predict) == correct_index[obs]:
                    TOP1 += 1
                
                if correct_index[obs] in (np.argsort(-predict)[:5]):
                    TOP5 += 1
                
                
        return TOP1, TOP5, TOP1/len(data), TOP5/len(data), len(data)
