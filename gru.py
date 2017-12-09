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

torch.manual_seed(42)

mode = "GRU"
difficulty = "easy"

if len(sys.argv) > 1:
	mode = sys.argv[1]

if len(sys.argv) > 2:
	difficulty = sys.argv[2]

if mode != "naive" and mode != "nlp-pre":
	mode = "GRU"

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

negative_words = [ "no", "never", "don't", "not", "nope", "negative", "can't, cant" ]

word_counter = Counter()

lemma = nltk.wordnet.WordNetLemmatizer()

stop_word_list = stopwords.words("english")

tokenizer = RegexpTokenizer(r'\w+')

force_use_all_dialogs = False
minimum_word_frequency = 5

verbosity_interval = 1000

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
			w2i[word] = len(w2i)

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
#Include loading of train and val data
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

#In forward pass first do an embedding of size 300
#Then loop through for each img_feature concatenate with fixed teacher
#then pass a layer on all the

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
	"""Evaluate a model on a batched data set."""
	TOP1 = 0.0
	TOP5 = 0.0

	for batch in minibatch(data, batch_size= batch_size):

		#Re-Initiate hidden state
		model.hidden = model.init_hidden()

		text_features, h5_ids, correct_index, last_words = preprocess(batch)
		lookup_text_tensor = Variable(torch.LongTensor([text_features])).squeeze()

		full_img_batch = np.empty([len(batch), output_vector_size, image_feature_size])
		for obs, img_ids in enumerate(h5_ids):
			for index, h5_id in enumerate(img_ids):
				full_img_batch[obs, index] = img_features[h5_id]

		full_img_batch = Variable(torch.from_numpy(full_img_batch).type(torch.FloatTensor))
		last_words = Variable(torch.LongTensor(last_words))

		prediction = model(lookup_text_tensor, full_img_batch, last_words)
		prediction = prediction.data.numpy()

		for obs, _ in enumerate(prediction):
			predict = prediction[obs]

			if np.argmax(predict) == correct_index[obs]:
			 	TOP1 += 1
			
			if correct_index[obs] in (np.argsort(-predict)[:5]):
				TOP5 += 1
			
			
	return TOP1, TOP5, TOP1/len(data), TOP5/len(data), len(data)


#Creates a class NaiveCBOW 
class GRU(nn.Module):
	#output_dim should be one according to the current dimention
	def __init__(self, vocab_size, embedding_dim, img_feature_size, output_dim, hidden_size, batch_size):
		super(GRU, self).__init__()
		#Parameters
		self.hidden_size = hidden_size
		self.output_vector_size = output_dim
		self.vocab_size = vocab_size
		self.batch_size = batch_size

		#Propagation layers
		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.gru = nn.GRU(embedding_dim, hidden_size, batch_first = True)
		self.hidden = self.init_hidden()
		self.linear = nn.Linear((hidden_size + img_feature_size), 1)
		self.activation_function = nn.Softmax()



		#Inputs should be in Tensor form
		#image_input is list of tensors
	def forward(self, text_input, image_input, last_words):
		embeds = self.embeddings(text_input)

		#Pass through GRU
		gru_out, self.hidden = self.gru(embeds, self.hidden)
		last_out = torch.gather(gru_out, 1, last_words.view(-1,1,1).expand(self.batch_size, 1, self.hidden_size) - 1)

		#Concatenate with pictures
		last_out_block = last_out.repeat(1, self.output_vector_size, 1)
		block = torch.cat((last_out_block, image_input), 2)

		score = self.linear(block).squeeze()
		score = self.activation_function(score)

		return score
		
	def init_hidden(self):
		return Variable(torch.zeros(self.batch_size, 1, self.hidden_size))
				


number_of_iterations = 1
learning_rate = 0.01
embedding_size = 300
image_feature_size = 2048
output_vector_size = 10
hidden_size = 100

model = GRU(nwords, embedding_size, image_feature_size, output_vector_size, hidden_size, batch_size)
print (model)

#TODO Check other learning rates
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
loss_data = []

#Initial scoring
#print("Initial Score on training", evaluate(model, train))
#print("Initial Score on development", evaluate(model, dev))

for ITER in range(number_of_iterations):

	random.shuffle(train)
	train_loss = 0.0
	start = time.time()
	iteration = 0

	for batch in minibatch(train, batch_size):

		model.zero_grad()
		optimizer.zero_grad()
		model.hidden = model.init_hidden()

		#Load data for model
		text_features, h5_ids, correct_index, last_words = preprocess(batch)
		lookup_text_tensor = Variable(torch.LongTensor([text_features])).squeeze()

		full_img_batch = np.empty([len(batch), output_vector_size, image_feature_size])

		for obs, img_ids in enumerate(h5_ids):
			for index, h5_id in enumerate(img_ids):
				full_img_batch[obs, index] = img_features[h5_id]
			
		full_img_batch = Variable(torch.from_numpy(full_img_batch).type(torch.FloatTensor))

		#Target
		target = Variable(torch.LongTensor([correct_index])).squeeze()
		#Vector for excluding padding effects
		last_words = Variable(torch.LongTensor(last_words))

		#Run model and calculate loss
		prediction = model(lookup_text_tensor, full_img_batch, last_words)
		loss = criterion(prediction, target)
		train_loss += loss.data[0]

		iteration += batch_size
		print (iteration)
		
		loss.backward()
		optimizer.step()

		#if iteration % verbosity_interval == 0 and iteration != 0:
		#	print("ITERATION %r: %r: train loss/sent=%.4f, time=%.2fs" % (ITER+1, iteration, train_loss/(iteration + 1), time.time() - start))
	
	print("ITERATION %r: train loss/sent=%.4f, time=%.2fs" % (ITER+1, train_loss/len(train), time.time() - start))
	print("Score on training", evaluate_batch(model, train))
	print("Score on development", evaluate_batch(model, dev))
	loss_data.append(train_loss/len(train))

#Save model
#torch.save(model, model_path + "/naive_cbow_easy.pty")
#print("Saved model has test score", evaluate(model, test))

#plot loss
plt.plot(loss_data)
plt.show()
