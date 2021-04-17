import pandas as pd
import csv
from data_aug import Augment
from steps import Steps
from nltk import RegexpTokenizer

col_list = ["id", "title", "author", "text", "label"]
train_set = pd.read_csv("fake-news/train.csv", usecols=col_list, encoding='UTF-8', delimiter=',')
data = train_set["text"]
data_labels = train_set["label"]
data_id = train_set["id"]
data_sample = data[0:500]
labels_sample = data_labels[0:500]
data_array = data_sample.to_numpy(dtype=str)
data_id_array = data_id.to_list()
data_label_array = data_labels.to_list()
data_list = data.to_list()[0:10000]
tokenizer = RegexpTokenizer(r'\w+')

data_no_punc = list(map(tokenizer.tokenize, data_array))
#for i in data_array:
#    data_no_punc.append(tokenizer.tokenize(i))

word_num = []
for i in data_no_punc:
    word_num.append(len(i))

tokenized_data = list(map(Steps.tokenize, data_array))
#for i in data_array:
#    tokenized_data.append(Steps.tokenize(i))

alpha_sr = [0, 0.05, 0.1, 0.15]
alpha_rd = [0, 0.05, 0.1, 0.15]
alpha_ri = [0, 0.05, 0.1, 0.15]
num_aug = [5, 10, 15]

for x in alpha_sr:
    for y in alpha_rd:
        for z in alpha_ri:
            if x == 0 and y == 0 and z == 0:
                continue
            else:
                for r in num_aug:
                    data_id_array_copy = data_id_array.copy()
                    data_list_copy = data_list.copy()
                    data_label_array_copy = data_label_array.copy()
                    augmented_data = []
                    augmented_data_labels = []
                    for i, j, k in zip(tokenized_data, word_num, labels_sample):
                        if j == 0:
                            continue
                        else:
                            augment = Augment(i, j, x, z, y, r)
                            augmented_data.extend(augment.augmented_sentences)
                            for h in range(len(augment.augmented_sentences)):
                                augmented_data_labels.append(k)

                    augmented_data_id = []
                    n = 20800
                    for i in range(len(augmented_data)):
                        augmented_data_id.append(n)
                        n = n + 1

                    data_list_copy.extend(augmented_data)
                    data_id_array_copy.extend(augmented_data_id)
                    data_label_array_copy.extend(augmented_data_labels)
                    rows = zip(data_id_array_copy, data_list_copy, data_label_array_copy)
                    csv_name = 'augmented_train(' + str(x) + ';' + str(y) + ';' + str(z) + ';' + str(r) + ')500;50%.csv'
                    with open(csv_name, mode='w', encoding='UTF-8', newline="") as n_file:
                        field_n = ['id', 'text', 'label']
                        writer = csv.writer(n_file)
                        writer.writerow(field_n)
                        for i in rows:
                            writer.writerow(i)
