import pandas as pd
from steps import Steps
from FeatureExtraction import Features
from nltk.tokenize import RegexpTokenizer
import csv
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Loading data
col_list_train = ["id", "text", "label"]
col_list_test = ["id", "text"]
train_set = pd.read_csv("Data_augmentation/augmented_train(0.1;0;0;5)500.csv", usecols=col_list_train,
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

# Measuring Flesch reading ease score for each data point
FRE = list(map(Features.flesch_re, data_array))

# Measuring Average words per sentence score for each data point
avg_WPS = list(map(Features.avg_word_per_sentence, data_array))

# Measuring Average syllables per word score for each data point
avg_SPW = list(map(Features.avg_syllables_PW, data_array))

# Measuring number of difficult words score for each data point
num_DW = list(map(Features.num_difficult_words, data_array))

# removing punctuation from text
data_no_punc = []
for i in data_array:
    j = i.lower()
    data_no_punc.append(tokenizer.tokenize(j))

# removing stop words from text
data_no_SW = list(map(Steps.remove_stop_words, data_no_punc))

# applying word vectorization
vectorizer = TfidfVectorizer(max_features=500)
vectors = vectorizer.fit_transform(data_array_v)
feature_names = vectorizer.get_feature_names()

vtz = CountVectorizer(vocabulary=feature_names, max_features=500)
n_vectors = vtz.transform(data_array).toarray()
# Converting word vectors produced into a list
word_vectors = list(map(list, n_vectors))

# Calculating the number of words for each data point
word_num = []
for i in data_no_punc:
    word_num.append(len(i))
# Tokenizing text
tokenized_data = list(map(Steps.tokenize, data_array))
# Measuring sentiment score for each data point
sentiment = list(map(Features.sentiment, data_no_punc))
# Applying pos-tagging to tokenized text
tagged_data = list(map(Steps.pos_tagging, tokenized_data))
# Measuring Formality score for each data pint
Formality_m = list(map(Features.formality, tagged_data, word_num))

# Zipping all relevant measures to be written into the csv file
rows = zip(data_id_array, word_vectors, Formality_m, FRE, sentiment, avg_WPS, avg_SPW, num_DW)
with open('Test(0.1;0;0;5)500.csv', mode='w', encoding='UTF-8', newline="") as n_file:
    field_n = ['id', 'word_vectors', 'Formality', 'FleschReadingEase', 'Sentiment', 'avg_words_per_sentence',
               'avg_syllables_per_word', 'number_of_difficult_words']
    writer = csv.writer(n_file)
    writer.writerow(field_n)
    for i in rows:
        writer.writerow(i)
