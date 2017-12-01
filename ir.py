#Import load data file
from data import Data
from naive_cbow import NaiveCBOW
from regression import Regression

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

mode = "rnn"
difficulty = "easy"
generative = False

if len(sys.argv) > 1:
    mode = sys.argv[1]

if len(sys.argv) > 2:
    difficulty = sys.argv[2]

if len(sys.argv) > 3:
    generative = sys.argv[3] == "g"

if mode != "naive" and mode != "nlp-pre":
    mode = "rnn"

if difficulty != "hard":
    difficulty = "easy"

if generative is None:
    generative = False

data_path = "data"
model_path = "models"

generative_label = "generative" if generative else "discriminative"
print("Starting 'Image Retrieval' in '" + mode + "' mode with '" + difficulty + "' data, using a '" + generative_label + "' model")
model_name = difficulty + "_" + mode + "_" + generative_label

data = Data(difficulty=difficulty, data_path=data_path)
(img_features, w2i, i2w, nwords, UNK, PAD) = data(preprocess=True)

#Creates a list with word indexes, list of all target image indexes, and tag_index 
train = list(data.get_train_data())
dev = list(data.get_validation_data())
test = list(data.get_test_data())

train = train[:2000]

number_of_iterations = 2
learning_rate = 0.005
embedding_size = 300
image_feature_size = 2048
output_vector_size = 10
verbosity_interval = 1000

if generative:
    model = Regression(nwords, embedding_size, image_feature_size, output_vector_size)
    criterion = nn.MSELoss()
else:
    model = NaiveCBOW(nwords, embedding_size, image_feature_size, output_vector_size)
    criterion = nn.CrossEntropyLoss()

evaluate = model.evaluate
print(model)

#TODO Check other learning rates
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loss_values = []

#Initial scoring
print("Score on training", evaluate(model, train, img_features))
print("Score on development", evaluate(model, dev, img_features))

for ITER in range(number_of_iterations):

    random.shuffle(train)
    train_loss = 0.0
    start = time.time()

    for iteration, (text_features, h5_ids, correct_index) in enumerate(train):

        #Data for the model
        lookup_text_tensor = Variable(torch.LongTensor([text_features]))
        lookup_img_list = [Variable(torch.from_numpy(img_features[h5_id])).view(1, -1) for h5_id in h5_ids]

        if generative:
            target = lookup_img_list[correct_index]
            prediction = model(lookup_text_tensor)
        else:
            target = Variable(torch.LongTensor([correct_index]))
            prediction = model(lookup_text_tensor, lookup_img_list)

        #Define loss
        loss = criterion(prediction, target)
        train_loss += loss.data[0]

        optimizer.zero_grad()
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % verbosity_interval == 0:
            print("ITERATION %r: %r: train loss/sent=%.4f, time=%.2fs" % (ITER+1, iteration, train_loss/(iteration + 1), time.time() - start))

    print("ITERATION %r: train loss/sent=%.4f, time=%.2fs" % (ITER+1, train_loss/len(train), time.time() - start))
    print("Score on training", evaluate(model, train, img_features))
    print("Score on development", evaluate(model, dev, img_features))
    train_loss_values.append(train_loss/len(train))

#Save model
torch.save(model, model_path + "/" + model_name + ".pty")
print("Saved model has test score", evaluate(model, test, img_features))

plt.plot(train_loss_values, label = "Train loss")
plt.legend(loc='best')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title (model_name + " - has loss with lr = %.4f, embedding size = %r" % (learning_rate, embedding_size))
plt.show()
