#Import load data file
from data import Data
from model import Regression
from evaluate import Evaluate

#Import other modules/libraries
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt

torch.manual_seed(42)

class Train():
    def __init__(self, difficulty):
        self.data_path = "../data"
        self.model_path = "../models"
        self.output_path = "../outputs"
        self.difficulty = difficulty
        self.timestamp = str(int(time.time()))
        self.model_name = "regression_" + self.difficulty
        self.data = Data(difficulty=self.difficulty, data_path=self.data_path)
        (self.img_features, self.w2i, self.i2w, self.nwords, self.UNK, self.PAD) = self.data()
        self.train = list(self.data.get_train_data())
        self.dev = list(self.data.get_validation_data())
        self.test = list(self.data.get_test_data())
        self.image_feature_size = 2048
        self.output_vector_size = 10

    def __call__(self, number_of_iterations=2, learning_rate=0.005, embedding_size=300):
        print("Starting 'Image Retrieval' in 'Regression' mode with '" + self.difficulty + "' data")

        self.model_full_path = self.model_path + "/" + self.model_name + "_" + self.timestamp + "_" + str(learning_rate) + "_" + str(embedding_size) + ".pty"
        self.output_file_name = self.output_path + "/" + self.model_name + "_" + self.timestamp + "_" + str(learning_rate) + "_" + str(embedding_size) + ".csv"

        self.number_of_iterations = number_of_iterations
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size

        self.model = Regression(self.nwords, self.embedding_size, self.image_feature_size, self.output_vector_size)
        self.criterion = nn.MSELoss()

        self.evaluate = Evaluate(self.model, self.img_features, self.minibatch, self.preprocess, self.image_feature_size)
        print(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.train_loss_values = []
        self.dev_loss_values = []
        self.test_loss_values = []

        self.magic()

        self.save_model()

        self.save_data()

    def minibatch(self, data, batch_size = 50):
        for i in range(0, len(data), batch_size):
            yield data[i:i+batch_size]

    def preprocess(self, batch):
        """Helper function for functional batches"""
        correct_indexes = [observation[2] for observation in batch]
        img_ids = [observation[1] for observation in batch]
        text_features = [observation[0] for observation in batch]

        #Add Padding to max len of sentence in batch
        max_length = max(map(len, text_features))
        text_features = [txt + [self.PAD] * (max_length - len(txt)) for txt in text_features]

        #return in "stacked" format 
        return text_features, img_ids, correct_indexes

    def magic(self):
        for ITER in range(self.number_of_iterations):

            random.shuffle(self.train)
            train_loss = 0.0
            start = time.time()

            for iteration, batch in enumerate(self.minibatch(self.train)):
                #Outputs matrices of batch size
                text_features, h5_ids, correct_index = self.preprocess(batch)
                lookup_text_tensor = Variable(torch.LongTensor([text_features])).squeeze()

                target = np.empty([len(batch), self.image_feature_size])
                for obs, img_ids in enumerate(h5_ids):
                    target[obs] = self.img_features[img_ids[correct_index[obs]]]

                target = Variable(torch.from_numpy(target).type(torch.FloatTensor))
                
                #Run model and calculate loss
                prediction = self.model(lookup_text_tensor)
                loss = self.criterion(prediction, target)
                train_loss += loss.data[0]

                self.optimizer.zero_grad()   
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

                #if iteration % verbosity_interval == 0:
                #    print("ITERATION %r: %r: train loss/sent=%.4f, time=%.2fs" % (ITER+1, iteration, train_loss/(iteration + 1), time.time() - start))

            print("ITERATION %r: train loss/sent=%.4f, time=%.2fs" % (ITER+1, train_loss/len(self.train), time.time() - start))
            #print("Score on training", evaluate(train))
            #print("Score on development", evaluate(dev))
            self.train_loss_values.append(train_loss/len(self.train))
            self.dev_loss_values.append(self.evaluate.calculate_loss(self.dev))
            self.test_loss_values.append(self.evaluate.calculate_loss(self.test))

    def save_model(self):
        #Save model
        torch.save(self.model, self.model_full_path)
        print("Saved model has test score", self.evaluate(self.test))

    def plot(self):
        plt.plot(self.train_loss_values, label = "Train loss")
        plt.plot(self.dev_loss_values, label = "Validation loss")
        plt.plot(self.test_loss_values, label = "Test loss")
        plt.legend(loc='best')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(self.model_name + " - has loss with lr = %.4f, embedding size = %r" % (self.learning_rate, self.embedding_size))
        plt.show()

    def save_data(self):
        file = open(self.output_file_name, "w")
        file.write(", ".join(map(str, self.train_loss_values)))
        file.write("\n")
        file.write(", ".join(map(str, self.dev_loss_values)))
        file.write("\n")
        file.write(", ".join(map(str, self.test_loss_values)))
        file.write("\n")
        file.write(str(self.evaluate(self.test)))
        file.write("\n")
        file.close()
