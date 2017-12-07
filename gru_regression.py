#Import load data file
import h5py
import json
import numpy as np
import nltk
from collections import defaultdict
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
#Import other modules/libraries
import sys
import time
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

torch.manual_seed(42)

mode = "regression"
difficulty = "easy"

if len(sys.argv) > 1:
	mode = sys.argv[1]

if len(sys.argv) > 2:
	difficulty = sys.argv[2]

if mode != "naive" and mode != "nlp-pre":
	mode = "regression"

#if difficulty != "hard":
#	difficulty = "easy"

data_path = "data"
model_path = "models"

print("Starting 'Image Retrieval' in '" + mode + "' mode with '" + difficulty + "' data")

#Load Image Features
path_to_h5_file = data_path + "/" + "/IR_image_features.h5"
path_to_json_file = data_path + "/" + "/IR_img_features2id.json"

img_features = np.asarray(h5py.File(path_to_h5_file, 'r')['img_features'])

with open(path_to_json_file, 'r') as f:
     visual_feat_mapping = json.load(f)['IR_imgid2id']

negative_words = [ "no", "never", "don't", "not", "nope", "negative", "can't" ]

word_counter = Counter()

lemma = nltk.wordnet.WordNetLemmatizer()

stop_word_list = stopwords.words("english")

tokenizer = RegexpTokenizer(r'\w+')

force_use_all_dialogs = False
minimum_word_frequency = 5


#Load Text Features
def read_dataset_text(filename):
	with open(filename, 'r') as f:
		dict_file = json.load(f)
		return dict_file

def create_w2i(dict_file):
	"""Creates w2i dictionary, excluding less common words, calls extract_words"""
	#loop through dataset - key is the datapoint number
	for key in dict_file.keys():
		word_list = extract_words(dict_file[key])
		for word in word_list:
			word_counter[word] += 1
	for (word, count) in word_counter.most_common():
		if count >= minimum_word_frequency and word not in w2i:
			i2w[w2i[word]] = word

def sentence_to_words(sentence):
	"""Tokenize a given sentence/string"""
	if mode == "naive":
		return sentence.split()
	return tokenizer.tokenize(sentence)

def extract_key_words(sentence):
	"""Function outputs relevant words for the model
	   it removes non meaningfull words"""


def extract_words(data_point):
	#Create word list for dialog
	#TODO pos_tags -> include only keywords
	word_list = sentence_to_words(data_point["caption"])
	#In naive mode, we include only the caption
	if mode == "naive":
		return word_list
	for dialog in data_point["dialog"]:
		for dialog_text in dialog:
			#Check whether dialog is valid
			if valid_dialog(dialog_text):
				#append to list new words from the sentence in dialog
				#TODO pos_tags -> include only keywords
				#Create dummy_sentence
				word_list.extend(sentence_to_words(dialog_text))
				
	word_list = [word for word in word_list if word not in stop_word_list]
	lowercase_word_list = [word.lower() for word in word_list]
	lemma_word_list = [lemma.lemmatize(word) for word in lowercase_word_list]
	#print ("original list", word_list[:100], "\n")
	#print ("lower_case list", lowercase_word_list[:100], "\n")
	#print ("lemma_list", lemma_word_list[:100])
	return lemma_word_list

def valid_dialog(dialog):
	"""Excludes negative questions and answers from dialog"""
	if force_use_all_dialogs:
		return True
	for negative_word in negative_words:
		if negative_word in dialog:
			return False
	return True

#Load Data
#Store caption vocabulary to a defaultdict
w2i = defaultdict(lambda: len(w2i))
UNK = w2i["<unk>"]
PAD = w2i["<pad>"]
i2w = dict()
i2w[w2i["<unk>"]] = "<unk>"
i2w[w2i["<pad>"]] = "<pad>"


train_file = data_path + "/" + difficulty + "/IR_train_" + difficulty + ".json"
dict_file_train = read_dataset_text(train_file)
create_w2i(dict_file_train)

validation_file = data_path + "/" + difficulty + "/IR_val_" + difficulty + ".json"
dict_file_validation = read_dataset_text(validation_file)
#create_w2i(dict_file_val)

test_file = data_path + "/" + difficulty + "/IR_test_" + difficulty + ".json"
dict_file_test = read_dataset_text(test_file)
#create_w2i(dict_file_val)


w2i = defaultdict(lambda:UNK, w2i)
nwords = len(w2i)
print("Vocabulary size is", len(w2i) - 2)

#creates a list of [list, list[10 np.array], correct index]
#creates a list with word indexes, list of all target image indexes, and tag_index
def get_data(dict_file):
	for key in dict_file.keys():
		word_list = extract_words(dict_file[key])
		tag_index = dict_file[key]["target"]
		img_list = dict_file[key]["img_list"]
		img_vector = [visual_feat_mapping[str(img_id)] for img_id in img_list]

		yield ([w2i[word] for word in word_list], img_vector, tag_index)


#Creates a list with word indexes, list of all target image indexes, and tag_index 
train = list(get_data(dict_file_train))
dev = list(get_data(dict_file_validation))
test = list(get_data(dict_file_test))

batch_size = 100

def minibatch(data, batch_size = 50):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def preprocess(batch):
	"""Helper function for functional batches"""
	correct_indexes = [observation[2] for observation in batch]
	img_ids = [observation[1] for observation in batch]
	text_features = [observation[0] for observation in batch]
	last_words = [len(dialog) for dialog in text_features]

	#Add Padding to max len of sentence in batch
	max_length = max(map(len, text_features))
	text_features = [txt + [PAD] * (max_length - len(txt)) for txt in text_features]

	#return in "stacked" format, added last_words for excluding padding effects on LSTM
	return text_features, img_ids, correct_indexes, last_words



def evaluate_batch(model, data):
	TOP1 = 0.0
	TOP5 = 0.0
	criterion = nn.MSELoss()


	for batch in minibatch(data, batch_size = batch_size):

		model.hidden = model.init_hidden()

		#Outputs matrices of batch size
		text_features, h5_ids, correct_index, last_words = preprocess(batch)

		lookup_text_tensor = Variable(torch.LongTensor([text_features])).squeeze()
		last_words = Variable(torch.LongTensor(last_words))

		scores = model(lookup_text_tensor, last_words)

		#create list of: batch size * 10 * 2048
		for obs, img_ids in enumerate (h5_ids):

			lookup_img_list = [Variable(torch.from_numpy(img_features[h5_id])).view(1,-1) for h5_id in img_ids]
			losses = np.empty([10,1])

			for index, single_image in enumerate(lookup_img_list):
				#Calculate loss per image
				loss = criterion(scores[obs], single_image)
				losses[index] = loss.data.numpy()[0]

			#Get TOP1 and TOP5
			if np.argmin(losses) == correct_index[obs]:
				TOP1 += 1

			if correct_index[obs] in np.argsort(losses, axis = 0)[:5, ]:
				TOP5 += 1

	return TOP1, TOP5, TOP1/len(data), TOP5/len(data), len(data)

def calculate_loss(model, data):

	c_loss = 0.0
	for batch in minibatch(data, batch_size = batch_size):

		model.hidden = model.init_hidden()

		text_features, h5_ids, correct_index, last_words = preprocess(batch)
		lookup_text_tensor = Variable(torch.LongTensor([text_features])).squeeze()

		target = np.empty([len(batch), image_feature_size])
		last_words = Variable(torch.LongTensor(last_words))

		for obs, img_ids in enumerate (h5_ids):
			target[obs] = img_features[img_ids[correct_index[obs]]]

		target = Variable(torch.from_numpy(target).type(torch.FloatTensor))
		prediction = model(lookup_text_tensor, last_words)
		loss = criterion(prediction, target)
		c_loss += loss.data[0] 

	return c_loss / len(data)


#Creates a class Regression 
class GRU_Regression(nn.Module):
	#output_dim should be one according to the current dimention
	def __init__(self, vocab_size, img_feature_size, embedding_dim, hidden_size, output_dim, batch_size):
		super(GRU_Regression, self).__init__()

		self.output_vector_size = output_dim
		self.vocab_size =  vocab_size
		self.batch_size = batch_size
		self.hidden_size = hidden_size

		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.hidden = self.init_hidden()
		self.gru = nn.GRU(embedding_dim, hidden_size, batch_first = True)
		self.linear = nn.Linear(hidden_size, img_feature_size)

   
		
	def forward(self, text_input, last_words):
		embeds = self.embeddings(text_input) 
		
		gru_out, self.hidden = self.gru(embeds, self.hidden)
		last_out = torch.gather(gru_out, 1, last_words.view(-1,1,1).expand(self.batch_size, 1, self.hidden_size) - 1)

		output = self.linear(last_out).squeeze()

		return output

	def init_hidden(self):
		return Variable(torch.zeros(1, 1, self.hidden_size))


number_of_iterations = 2
learning_rate = 0.005
embedding_size = 300
image_feature_size = 2048
output_vector_size = 10
hidden_size = 500

model = GRU_Regression(nwords, image_feature_size, embedding_size, hidden_size, output_vector_size, batch_size)
print (model)

#TODO Check other learning rates
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.MSELoss()
train_loss_list = []
dev_loss = []
test_loss = []

#Initial scoring
#print("Initial Score on training", evaluate_batch(model, train))
#print("Initial Score on development", evaluate_batch(model, dev))
#print("Initial Score on training", evaluate(model, train))
#print("Initial Score on development", evaluate(model, dev))

for ITER in range(number_of_iterations):

	random.shuffle(train)
	train_loss = 0.0
	start = time.time()
	iteration = 0

	for batch in minibatch(train, batch_size):

		#Re-Initiate hidden state
		model.hidden = model.init_hidden()
		optimizer.zero_grad()	
		model.zero_grad()

		#Outputs matrices of batch size
		text_features, h5_ids, correct_index, last_words = preprocess(batch)

		lookup_text_tensor = Variable(torch.LongTensor([text_features])).squeeze()
		last_words = Variable(torch.LongTensor(last_words))

		target = np.empty([len(batch), image_feature_size])
		for obs, img_ids in enumerate (h5_ids):
			target[obs] = img_features[img_ids[correct_index[obs]]]

		target = Variable(torch.from_numpy(target).type(torch.FloatTensor))
		
		#Run model and calculate loss
		prediction = model(lookup_text_tensor, last_words)
		loss = criterion(prediction, target)
		train_loss += loss.data[0]
		
		loss.backward()
		optimizer.step()

		iteration += batch_size
		print (iteration)

	
	print("ITERATION %r: train loss/sent=%.4f, time=%.2fs" % (ITER+1, train_loss / len(train), time.time() - start))
	#print("Score on training", evaluate_batch(model, train))
	#print("Score on development", evaluate_batch(model, dev))

	train_loss_list.append(train_loss/len(train))
	dev_loss.append(calculate_loss(model, dev))
	test_loss.append(calculate_loss(model, test))

print("Saved model has test score", evaluate_batch(model, test))

#plot loss
plt.plot(train_loss_list, label = "Train loss")
plt.plot(dev_loss, label = "Validation loss")
plt.plot(test_loss, label = "Test loss")
plt.legend(loc='best')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title ("Loss with lr = %.4f, embedding size = %r" % (learning_rate, embedding_size))

plt.show()

#Save model
#torch.save(model, model_path + "/regression_easy.pt")
#torch.save(model.state_dict(), model_path + "/regression_easy_dict.pty")


