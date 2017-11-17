import h5py
import json
import numpy as np
from collections import defaultdict

#Load Image Features
path_to_h5_file = "Data/IR_image_features.h5"
path_to_json_file = "Data/IR_img_features2id.json"

img_features = np.asarray(h5py.File(path_to_h5_file, 'r')['img_features'])

with open(path_to_json_file, 'r') as f:
     visual_feat_mapping = json.load(f)['IR_imgid2id']


#Load Text Features
def read_dataset_text(filename):
	with open(filename, 'r') as f:
		dict_file = json.load(f)
		return dict_file

def create_w2i(dict_file):
	for key in dict_file.keys():
		caption = dict_file[key]["caption"].split()
		for word in caption:
			if word not in w2i:
				w2i[word] = len(w2i)

#Load Data
#Include loading of train and val data
#Store caption vocabulary to a defaultdict
w2i = defaultdict(lambda: len(w2i))

t_file = "Data/Easy/IR_train_easy.json" 
dict_file_train = read_dataset_text(t_file)
create_w2i(dict_file_train)

t_file = "Data/Easy/IR_val_easy.json"
dict_file_val = read_dataset_text(t_file)
create_w2i(dict_file_val)

t_file = "Data/Easy/IR_test_easy.json"
dict_file_test = read_dataset_text(t_file)
create_w2i(dict_file_val)

UNK = w2i["<unk>"]
w2i = defaultdict(lambda: UNK, w2i)
nwords = len(w2i)

#####################################
#Visualize caption
#for key in dict_file.keys():
#	print (dict_file[key]["caption"])
#	print(dict_file[key]["target_img_id"])
#	print ("\n")
#####################################


#creates a list of [list, list[10 np.array], correct index]
#creates a list with word indexes, list of all target image indexes, and tag_index
def data_naive_cbow(dict_file):
	for key in dict_file.keys():
		caption = dict_file[key]["caption"].split()
		tag_index = dict_file[key]["target"]
		img_list = dict_file[key]["img_list"]
		img_vector = [visual_feat_mapping[str(img_id)] for img_id in img_list]

		yield ([w2i[word] for word in caption], img_vector, tag_index)







#In forward pass first do an embedding of size 300
#Then loop through for each img_feature concatenate with fixed teacher
#then pass a layer on all the 




