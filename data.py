import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import numpy as np
import h5py
import json
from collections import Counter
from collections import defaultdict

class Data:
    def __init__(self, difficulty="easy", data_path="data"):
        self.path_to_h5_file = data_path + "/IR_image_features.h5"
        self.path_to_json_file = data_path + "/IR_img_features2id.json"

        self.train_file = data_path + "/" + difficulty + "/IR_train_" + difficulty + ".json"
        self.validation_file = data_path + "/" + difficulty + "/IR_val_" + difficulty + ".json"
        self.test_file = data_path + "/" + difficulty + "/IR_test_" + difficulty + ".json"

        self.negative_words = [ "no", "never", "don't", "not", "nope", "negative", "can't" ]
        self.lemma = nltk.wordnet.WordNetLemmatizer()
        self.stop_word_list = stopwords.words("english")
        self.tokenizer = RegexpTokenizer(r"\w+")

        self.img_features = np.asarray(h5py.File(self.path_to_h5_file, "r")["img_features"])

        with open(self.path_to_json_file, "r") as f:
            visual_feat_mapping = json.load(f)["IR_imgid2id"]

        self.visual_feat_mapping = visual_feat_mapping

    def __call__(self, preprocess=True, force_use_all_dialogs=False):
        self.preprocess = preprocess
        self.force_use_all_dialogs = force_use_all_dialogs
        self.dict_file_train = self.read_dataset_text(self.train_file)
        self.dict_file_validation = self.read_dataset_text(self.validation_file)
        self.dict_file_test = self.read_dataset_text(self.test_file)

        (self.w2i, self.i2w, self.nwords) = self.create_w2i(self.dict_file_train)

        print("Vocabulary size is", len(self.w2i))

        return (self.w2i, self.i2w, self.nwords)

    def get_train_data(self):
        return self.get_data(self.dict_file_train)

    def get_validation_data(self):
        return self.get_data(self.dict_file_validation)

    def get_test_data(self):
        return self.get_data(self.dict_file_test)

    def read_dataset_text(self, filename):
        with open(filename, "r") as f:
            dict_file = json.load(f)
            return dict_file

    def create_w2i(self, dict_file, minimum_word_frequency=5):
        word_counter = Counter()
        w2i = defaultdict(lambda: len(w2i))
        UNK = w2i["<unk>"]
        PAD = w2i["<pad>"]
        i2w = dict()
        i2w[UNK] = "<unk>"
        i2w[PAD] = "<pad>"
        for key in dict_file.keys():
            word_list = self.extract_words(dict_file[key])
            for word in word_list:
                word_counter[word] += 1
        for (word, count) in word_counter.most_common():
            if word not in w2i:
                if (self.preprocess and count >= minimum_word_frequency) or not self.preprocess:
                    i2w[w2i[word]] = word
        w2i = defaultdict(lambda: UNK, w2i)
        nwords = len(w2i)
        return (w2i, i2w, nwords)

    def sentence_to_words(self, sentence):
        if self.preprocess:
            return self.tokenizer.tokenize(sentence)
        else:
            return sentence.split()

    def extract_words(self, data_point):
        word_list = self.sentence_to_words(data_point["caption"])
        if not self.preprocess:
            return word_list
        for dialog in data_point["dialog"]:
            for dialog_text in dialog:
                if self.valid_dialog(dialog_text):
                    word_list.extend(self.sentence_to_words(dialog_text))
        word_list = [word for word in word_list if word not in self.stop_word_list]
        lowercase_word_list = [word.lower() for word in word_list]
        lemma_word_list = [self.lemma.lemmatize(word) for word in lowercase_word_list]
        return lemma_word_list

    def valid_dialog(self, dialog):
        if self.force_use_all_dialogs:
            return True
        for negative_word in self.negative_words:
            if negative_word in dialog:
                return False
        return True

    def get_data(self, dict_file):
        for key in dict_file.keys():
            word_list = self.extract_words(dict_file[key])
            tag_index = dict_file[key]["target"]
            img_list = dict_file[key]["img_list"]
            img_vector = [self.visual_feat_mapping[str(img_id)] for img_id in img_list]

            yield ([self.w2i[word] for word in word_list], img_vector, tag_index)

#data = Data(difficulty="easy", data_path="data")
#print(data(preprocess=True))
#print(list(data.get_test_data())[:10])

