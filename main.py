import pandas as pd
from steps import Steps
from FeatureExtraction import Features
from nltk.tokenize import RegexpTokenizer
import csv
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

col_list_train = ["id", "text", "label"]
col_list_test = ["id", "text"]
train_set = pd.read_csv("Data_augmentation/augmented_train(0.05;0;0;15)500;50%.csv", usecols=col_list_train,
                        encoding='UTF-8', delimiter=',')
test_set = pd.read_csv("fake-news/test.csv", usecols=col_list_test, encoding='UTF-8', delimiter=',')
data = test_set["text"]
data_id = test_set["id"]
train_data = train_set["text"]
#data_label = train_set["label"]
data_sample = data
data_array = data_sample.to_numpy(dtype=str)
data_array_v = train_data.to_numpy(dtype=str)
data_id_array = data_id.to_list()
#data_label_array = data_label.to_list()
tokenizer = RegexpTokenizer(r'\w+')

FRE = list(map(Features.flesch_re, data_array))

avg_WPS = list(map(Features.avg_word_per_sentence, data_array))

avg_SPW = list(map(Features.avg_syllables_PW, data_array))

num_DW = list(map(Features.num_difficult_words, data_array))

data_no_punc = []
for i in data_array:
    j = i.lower()
    data_no_punc.append(tokenizer.tokenize(j))

data_no_SW = list(map(Steps.remove_stop_words, data_no_punc))

vectorizer = TfidfVectorizer(max_features=500)
vectors = vectorizer.fit_transform(data_array_v)
feature_names = vectorizer.get_feature_names()

vtz = CountVectorizer(vocabulary=feature_names, max_features=500)
n_vectors = vtz.transform(data_array).toarray()

word_vectors = list(map(list, n_vectors))

word_num = []
for i in data_no_punc:
    word_num.append(len(i))

tokenized_data = list(map(Steps.tokenize, data_array))

sentiment = list(map(Features.sentiment, data_no_punc))

tagged_data = list(map(Steps.pos_tagging, tokenized_data))

Formality_m = list(map(Features.formality, tagged_data, word_num))

rows = zip(data_id_array, word_vectors, Formality_m, FRE, sentiment, avg_WPS, avg_SPW, num_DW,)
with open('Test5%15;50%.csv', mode='w', encoding='UTF-8', newline="") as n_file:
    field_n = ['id', 'word_vectors', 'Formality', 'FleschReadingEase', 'Sentiment', 'avg_words_per_sentence',
               'avg_syllables_per_word', 'number_of_difficult_words']
    writer = csv.writer(n_file)
    writer.writerow(field_n)
    for i in rows:
        writer.writerow(i)
