#Import load data file
from load_all_data import *
#Import other modules/libraries
import time
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
torch.manual_seed(42)

#Creates a list with word indexes, list of all target image indexes, and tag_index 
train = list(data_naive_cbow(dict_file_train))
dev = list(data_naive_cbow(dict_file_val))
test = list(data_naive_cbow(dict_file_test))



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

		if correct_index in np.argsort(-scores.data.numpy())[0,:5]:
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

		#Inputs should be in Tensor form
		#image_input is list of tensors
	def forward(self, text_input, image_input):
		embeds = self.embeddings(text_input)
		sum_embeds = torch.sum(embeds, 1)
		#create block matrix
		sum_embeds_block = sum_embeds.repeat(10,1)
		image_block = torch.stack(image_input).squeeze()
		block = torch.cat((sum_embeds_block, image_block), 1)
		
		score = self.linear(block)
		output = self.activation_function(score.view(1,-1))
		
		return output


model = NaiveCBOW(nwords, 300, 2048, 10)


#TODO Check other learning rates
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss = nn.CrossEntropyLoss()

#First scoring on validation
print ("Score on validation", evaluate(model, dev))

for ITER in range(10):

	random.shuffle(train)
	train_loss = 0.0
	start = time.time()

	for iteration, (text_features, h5_ids, correct_index) in enumerate(train[:2000]):

		#Data for the model
		lookup_text_tensor = Variable(torch.LongTensor([text_features]))
		lookup_img_list = [Variable(torch.from_numpy(img_features[h5_id])).view(1, -1) for h5_id in h5_ids]
		score = model(lookup_text_tensor, lookup_img_list)
		target = Variable(torch.LongTensor([correct_index]))

		#Define loss
		output = loss(score, target)
		train_loss += output.data[0]

		model.zero_grad()
		output.backward()
		#print (model.linear.grad)
		optimizer.step()

		if iteration % 100 == 0:
			print("iter %r: %r: train loss/sent=%.4f, time=%.2fs" % (ITER, iteration, train_loss/(iteration + 1), time.time() - start))
	
	print("iter %r: train loss/sent=%.4f, time=%.2fs" % (ITER, train_loss/len(train), time.time()-start))
	print ("Score on validation", evaluate(model, dev))


#Save model
torch.save(model, "Models/naive_cbow_easy.pty")








