import spacy
import json
from collections import Counter
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import h5py
import json
import numpy as np
from thinc.describe import output
from torch.autograd import Variable
import tensorflow as tf
import sys
import pickle
import os.path
import time

torch.manual_seed(1)
nlp = spacy.load('en')

with open('data/Data/Easy/train_easy.json') as json_data:
    d = json.load(json_data)
print("Complete")


counter=Counter()
if(os.path.exists('dict.pickle')==False):
  for key in d.keys():
    dialog_arr=d[key]['dialog']
    caption=d[key]['caption']
    for dialog in dialog_arr:
        document=nlp(dialog[0].lower())
        for token in document:
            if token.tag_ =='NN' or token.tag_=='JJ' or (token.tag_).startswith( 'V' ):
                counter[token.text] +=1
  print("In Process")
  pickle_out = open("dict.pickle","wb")
  pickle.dump(counter, pickle_out)
  pickle_out.close()
else:
    pickle_in = open("dict.pickle", "rb")
    counter = pickle.load(pickle_in)

del counter['is']
del counter['are']
del counter['do']
counter=counter.most_common()

w2i={}
i2w={}
cnt=0
for data in counter:
    w2i[cnt]=data[0]
    i2w[data[0]]=cnt
    cnt+=1
w2i[cnt]="OTH"
i2w["OTH"]=cnt

embedding=300
w2i_embed={}
i2w_embed={}
count=0
for key in w2i.keys():
    if(count<embedding-1):
        val=w2i[key]
        w2i_embed[key]=val
        i2w_embed[val]=key
        count+=1
w2i_embed[cnt]="OTH"
i2w_embed["OTH"]=cnt

img_features = np.asarray(h5py.File('data/IR_image_features.h5', 'r')['img_features'])
visual_feat_mapping={}
with open('data/IR_img_features2id.json', 'r') as f:
     visual_feat_mapping = json.load(f)['IR_imgid2id']
print("Done")
#h5_id = visual_feat_mapping[str(img_id)]
#img_feat = img_features[h5_id]

class Gru:
    layers = 10
    sigmoid = nn.Sigmoid()
    tanh = nn.Tanh()
    batch_size=64
    #loss = nn.CosineEmbeddingLoss()
    # loss=nn.MSELoss()
    loss=nn.CrossEntropyLoss()
    one_mat = Variable(torch.ones(1, layers).cuda(), requires_grad=False)
    y = Variable(torch.ones([2048]).cuda())
    learning_rate = 0.001
    input_size=2048+embedding
    valid_count=0

    def __init__(self, embedding_size):
        self.z_u = Variable(torch.randn(embedding_size, Gru.layers).cuda(), requires_grad=True)
        self.z_w = Variable(torch.randn(Gru.layers, Gru.layers).cuda(), requires_grad=True)
        self.r_u = Variable(torch.randn(embedding_size, Gru.layers).cuda(), requires_grad=True)
        self.r_w = Variable(torch.randn(Gru.layers, Gru.layers).cuda(), requires_grad=True)
        self.h_u = Variable(torch.randn(embedding_size, Gru.layers).cuda(), requires_grad=True)
        self.h_w = Variable(torch.randn(Gru.layers, Gru.layers).cuda(), requires_grad=True)
        self.w_out = Variable(torch.randn(Gru.layers, Gru.layers).cuda(), requires_grad=True)
        self.optimizer = optim.Adagrad([self.z_u, self.z_w, self.r_u, self.r_w, self.h_u, self.h_w, self.w_out])


    def create_optimizer(self):
        self.optimizer = optim.Adagrad([self.z_u, self.z_w, self.r_u, self.r_w, self.h_u, self.h_w, self.w_out])

    def forward(self,network_data_list):
        m=nn.Softmax()
        output_data_list = []
        target_image = []
        index = 0
        image_index=0
        output_tensor=Variable(torch.zeros(Gru.batch_size,10).cuda(),requires_grad=True)
        input_images_tensor=Variable((torch.LongTensor(Gru.batch_size).zero_()).cuda())
        input_vectors = Variable(torch.zeros(Gru.batch_size,1,10).cuda(), requires_grad=False)
        sent_dict = {}
        for i in range (0,Gru.batch_size):

            data=network_data_list[i]
            input_images_tensor[i].data[0]=data.get_target_image_id()

            for img in data.get_image_data():

                h5_id = visual_feat_mapping[str(img)]
                img_feat = img_features[h5_id]

                img_vector = Variable((torch.from_numpy(img_feat).cuda()).view(2048, 1), requires_grad=False)
                s_prev = Variable(torch.zeros(1, Gru.layers).cuda(), requires_grad=True)
                s = Variable(torch.zeros(1, Gru.layers).cuda(), requires_grad=True)

                for sentence in data.get_input_data():
                    if((sentence in sent_dict.keys())==False):
                       document = nlp(sentence.lower())
                       imp_words = []

                       for token in document:
                          if (token.tag_ == 'NN' or token.tag_ == 'JJ' or (token.tag_).startswith('V')):
                              imp_words.append(token.text)

                       sent_vector = Variable(torch.zeros(embedding, 1).cuda(), requires_grad=False)
                       for word in imp_words:
                           if word in w2i_embed:
                              index = w2i[word]
                              sent_vector.data[index, 0] = 1
                           else:
                              sent_vector.data[embedding - 1, 0] = 1.0
                       sent_dict[sentence]=sent_vector
                    else:
                        sent_vector=sent_dict[sentence]

                    input_vector = torch.cat([img_vector, sent_vector])
                    input_vector = input_vector.view(1, 2348)

                    z_layer = Gru.sigmoid(input_vector.mm(self.z_u) + s_prev.mm(self.z_w))
                    r_layer = Gru.sigmoid(input_vector.mm(self.r_u) + s_prev.mm(self.r_w))
                    h_layer = Gru.tanh(input_vector.mm(self.h_u) + (s_prev * r_layer).mm(self.h_w))
                    s = (Gru.one_mat - z_layer) * h_layer + s_prev * z_layer
                    s_prev = s

                input_vectors[i]=s
        input_vectors=input_vectors.view(Gru.batch_size,Gru.layers)
        output=input_vectors.mm(self.w_out)
        loss = Gru.loss(output, input_images_tensor)
        return loss

    def valid_out(self,arr,index):
        #print(arr.data)
        #print(index)
        highest_val=arr.data[index][0]
        for i in range(0,len(arr.data)):
            if(arr.data[i][0]>highest_val):
                return False
        return True

    def softmax(self,output_data):
        sum=0.0
        for i in range(0,10):
            sum+=np.exp(output_data.data[i,0])
        for i in range(0,10):
            output_data.data[i,0]=np.exp(output_data.data[i,0])/sum
        return output_data

    def backward_propgation(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


class NetworkData:

    def __init__(self,input_text_data,image_data,target_image_id):
        self.input_text_data=input_text_data
        self.image_data=image_data
        self.target_image_id=target_image_id

    def get_input_data(self):
        return self.input_text_data

    def get_image_data(self):
        return self.image_data

    def get_target_image_id(self):
        return self.target_image_id



class TrainNetwork:

    def train_data(self, list_keys,text_data, image_data, gru):
        error = 0.0
        index = 0
        keys=[]
        for epoch in range(0, 7):
            for i in range(10000,20000,Gru.batch_size):
                index = index + 1
                network_data_list=[]
                for j in range(i,i+Gru.batch_size):
                    key=list_keys[j]
                    input_text_data = []
                    dialogs = text_data[key]['dialog']
                    image_data = text_data[key]['img_list']
                    caption_data = text_data[key]['caption']
                    target_image_id = text_data[key]['target_img_id']
                    target_index = text_data[key]['target']
                    input_text_data.append(caption_data)
                    for dialog in dialogs:
                        input_text_data.append(dialog[0])
                    network_data=NetworkData(input_text_data,image_data,target_index)
                    network_data_list.append(network_data)
                #start_time=time.clock()
                loss = gru.forward(network_data_list)
                #print((time.clock()-start_time))

                gru.backward_propgation(loss)
                error += loss.data[0]

                #print(error)
            print(error)
            error = 0.0
            index=0
            pickle_out = open("gru.pickle", "wb")
            pickle.dump(gru, pickle_out)


    def get_prediction(self,text_data,image_data,gru):
        li=[]
        for key in text_data.keys():
            li.append(key)
        for i in range(0,1000):
            key=li[i]
            input_text_data = []
            dialogs = text_data[key]['dialog']
            image_data = text_data[key]['img_list']
            caption_data = text_data[key]['caption']
            target_image_id = text_data[key]['target_img_id']
            target_index = text_data[key]['target']
            input_text_data.append(caption_data)
            for dialog in dialogs:
                input_text_data.append(dialog[0])

            gru.forward(input_text_data, image_data, target_image_id)
        print(Gru.valid_count)

if(os.path.exists('gru.pickle')==False):
   gru = Gru(2348)
else:
    pickle_in = open("gru.pickle", "rb")
    gru = pickle.load(pickle_in)
    #gru.create_optimizer()
list_keys=[]
for key in d.keys():
    list_keys.append(key)
train_network = TrainNetwork()
train_network.train_data(list_keys, d,visual_feat_mapping, gru)
#train_network.get_prediction(d, visual_feat_mapping, gru)