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

torch.manual_seed(42)

mode = "rnn"
difficulty = "easy"

if len(sys.argv) > 1:
	mode = sys.argv[1]

if len(sys.argv) > 2:
	difficulty = sys.argv[2]

if mode != "naive" and mode != "nlp-pre":
	mode = "rnn"

if difficulty != "hard":
	difficulty = "easy"

data_path = "data"
model_path = "models"

print("Starting 'Image Retrieval' in '" + mode + "' mode with '" + difficulty + "' data")

#Load Image Features
path_to_h5_file = data_path + "/" + difficulty + "/IR_image_features.h5"
path_to_json_file = data_path + "/" + difficulty + "/IR_img_features2id.json"

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

verbosity_interval = 1000

#Load Text Features
def read_dataset_text(filename):
	with open(filename, 'r') as f:
		dict_file = json.load(f)
		return dict_file

def create_w2i(dict_file):
	for key in dict_file.keys():
		word_list = extract_words(dict_file[key])
		for word in word_list:
			word_counter[word] += 1
	for (word, count) in word_counter.most_common():
		if count >= minimum_word_frequency and word not in w2i:
			w2i[word] = len(w2i)

def sentence_to_words(sentence):
	if mode == "naive":
		return sentence.split()
	return tokenizer.tokenize(sentence)

def extract_words(data_point):
	word_list = sentence_to_words(data_point["caption"])
	if mode == "naive":
		return word_list
	for dialog in data_point["dialog"]:
		for dialog_text in dialog:
			if valid_dialog(dialog_text):
				word_list.extend(sentence_to_words(dialog_text))
	word_list = [word for word in word_list if word not in stop_word_list]
	lowercase_word_list = [word.lower() for word in word_list]
	lemma_word_list = [lemma.lemmatize(word) for word in lowercase_word_list]
	return lemma_word_list

def valid_dialog(dialog):
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

train_file = data_path + "/" + difficulty + "/IR_train_easy.json"
dict_file_train = read_dataset_text(train_file)
create_w2i(dict_file_train)

validation_file = data_path + "/" + difficulty + "/IR_val_easy.json"
dict_file_validation = read_dataset_text(validation_file)
#create_w2i(dict_file_val)

test_file = data_path + "/" + difficulty + "/IR_test_easy.json"
dict_file_test = read_dataset_text(test_file)
#create_w2i(dict_file_val)

UNK = w2i["<unk>"]
w2i = defaultdict(lambda: UNK, w2i)
nwords = len(w2i)

print("Vocabulary size is", len(w2i))

#####################################
#Visualize caption
#for key in dict_file.keys():
#	print (dict_file[key]["caption"])
#	print(dict_file[key]["target_img_id"])
#	print ("\n")
#####################################

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

train = train[:2000]

number_of_iterations = 10
learning_rate = 0.01
embedding_size = 300
image_feature_size = 2048
output_vector_size = 10

def evaluate(model, data):
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

#Creates a class NaiveCBOW 
class NaiveCBOW(nn.Module):
	#output_dim should be one according to the current dimention
	def __init__(self, vocab_size, embedding_dim, img_feature_size, output_dim):
		super(NaiveCBOW, self).__init__()
		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		#TODO check if nn.Parameter or nn.Linear
		self.linear = nn.Linear((embedding_dim + img_feature_size), 1)
		self.activation_function = nn.Softmax()
		self.output_vector_size = output_dim

		#Inputs should be in Tensor form
		#image_input is list of tensors
	def forward(self, text_input, image_input):
		embeds = self.embeddings(text_input)
		sum_embeds = torch.sum(embeds, 1)
		#create block matrix
		sum_embeds_block = sum_embeds.repeat(self.output_vector_size, 1)
		image_block = torch.stack(image_input).squeeze()
		block = torch.cat((sum_embeds_block, image_block), 1)
		
		score = self.linear(block)
		output = self.activation_function(score.view(1, -1))

		#val, idx = torch.max(output, 1)

		return output

model = NaiveCBOW(nwords, embedding_size, image_feature_size, output_vector_size)

#TODO Check other learning rates
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

#Initial scoring
print("Score on training", evaluate(model, train))
print("Score on development", evaluate(model, dev))

for ITER in range(number_of_iterations):

	random.shuffle(train)
	train_loss = 0.0
	start = time.time()

	for iteration, (text_features, h5_ids, correct_index) in enumerate(train):

		#Data for the model
		lookup_text_tensor = Variable(torch.LongTensor([text_features]))
		lookup_img_list = [Variable(torch.from_numpy(img_features[h5_id])).view(1, -1) for h5_id in h5_ids]

		target = Variable(torch.LongTensor([correct_index]))

		optimizer.zero_grad()

		#print("target", target)

		prediction = model(lookup_text_tensor, lookup_img_list)

		#print("prediction", prediction)

		#Define loss
		loss = criterion(prediction, target)
		#print(loss.data[0])
		train_loss += loss.data[0]

		model.zero_grad()
		loss.backward()
		#print (model.linear.grad)
		optimizer.step()

		if iteration % verbosity_interval == 0:
			print("ITERATION %r: %r: train loss/sent=%.4f, time=%.2fs" % (ITER+1, iteration, train_loss/(iteration + 1), time.time() - start))
	
	print("ITERATION %r: train loss/sent=%.4f, time=%.2fs" % (ITER+1, train_loss/len(train), time.time() - start))
	print("Score on training", evaluate(model, train))
	print("Score on development", evaluate(model, dev))

#Save model
torch.save(model, model_path + "/naive_cbow_easy.pty")
print("Saved model has test score", evaluate(model, test))
