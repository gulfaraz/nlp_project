#Import load data file
from data import Data
from model import Regression
from evaluate import Evaluate

#Import other modules/libraries
import sys
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

difficulty = "easy"

if len(sys.argv) > 1:
    difficulty = sys.argv[1]

if difficulty != "hard":
    difficulty = "easy"

data_path = "../data"
model_path = "../models"

print("Starting 'Image Retrieval' in 'Regression' mode with '" + difficulty + "' data")
model_name = "regression_" + difficulty

data = Data(difficulty=difficulty, data_path=data_path)
(img_features, w2i, i2w, nwords, UNK, PAD) = data()

#Creates a list with word indexes, list of all target image indexes, and tag_index 
train = list(data.get_train_data())
dev = list(data.get_validation_data())
test = list(data.get_test_data())

def minibatch(data, batch_size = 50):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def preprocess(batch):
    """Helper function for functional batches"""
    correct_indexes = [observation[2] for observation in batch]
    img_ids = [observation[1] for observation in batch]
    text_features = [observation[0] for observation in batch]

    #Add Padding to max len of sentence in batch
    max_length = max(map(len, text_features))
    text_features = [txt + [PAD] * (max_length - len(txt)) for txt in text_features]

    #return in "stacked" format 
    return text_features, img_ids, correct_indexes

train = train

number_of_iterations = 10
learning_rate = 0.005
embedding_size = 300
image_feature_size = 2048
output_vector_size = 10
verbosity_interval = 1000

model = Regression(nwords, embedding_size, image_feature_size, output_vector_size)
criterion = nn.MSELoss()

evaluate = Evaluate(model, img_features, minibatch, preprocess, image_feature_size)
print(model)

#TODO Check other learning rates
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loss_values = []
dev_loss_values = []
test_loss_values = []

#Initial scoring
print("Score on training", evaluate(train))
print("Score on development", evaluate(dev))

for ITER in range(number_of_iterations):

    random.shuffle(train)
    train_loss = 0.0
    start = time.time()

    for iteration, batch in enumerate(minibatch(train)):
        #Outputs matrices of batch size
        text_features, h5_ids, correct_index = preprocess(batch)
        lookup_text_tensor = Variable(torch.LongTensor([text_features])).squeeze()

        target = np.empty([len(batch), image_feature_size])
        for obs, img_ids in enumerate (h5_ids):
            target[obs] = img_features[img_ids[correct_index[obs]]]

        target = Variable(torch.from_numpy(target).type(torch.FloatTensor))
        
        #Run model and calculate loss
        prediction = model(lookup_text_tensor)
        loss = criterion(prediction, target)
        train_loss += loss.data[0]

        optimizer.zero_grad()   
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % verbosity_interval == 0:
            print("ITERATION %r: %r: train loss/sent=%.4f, time=%.2fs" % (ITER+1, iteration, train_loss/(iteration + 1), time.time() - start))

    print("ITERATION %r: train loss/sent=%.4f, time=%.2fs" % (ITER+1, train_loss/len(train), time.time() - start))
    print("Score on training", evaluate(train))
    print("Score on development", evaluate(dev))
    train_loss_values.append(train_loss/len(train))
    dev_loss_values.append(evaluate.calculate_loss(dev))
    test_loss_values.append(evaluate.calculate_loss(test))

#Save model
torch.save(model, model_path + "/" + model_name + ".pty")
print("Saved model has test score", evaluate(test))

plt.plot(train_loss_values, label = "Train loss")
plt.plot(dev_loss_values, label = "Validation loss")
plt.plot(test_loss_values, label = "Test loss")
plt.legend(loc='best')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title (model_name + " - has loss with lr = %.4f, embedding size = %r" % (learning_rate, embedding_size))
plt.show()
