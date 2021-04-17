import tensorflow as t
import pandas as pd
# import math
import numpy as np


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

t.compat.v1.disable_eager_execution()


def normalize(array):
    return (array - array.mean()) / array.std()


def load_data():
    col_list_train = ['id', 'word_vectors', 'Formality', 'FleschReadingEase', 'Sentiment', 'avg_words_per_sentence',
                      'avg_syllables_per_word', 'number_of_difficult_words', 'label']
    col_list_test = ['id', 'word_vectors', 'Formality', 'FleschReadingEase', 'Sentiment', 'avg_words_per_sentence',
                     'avg_syllables_per_word', 'number_of_difficult_words']
    col_list_test_labels = ["id", "label"]

    train_set = pd.read_csv("Train.csv", usecols=col_list_train, encoding='UTF-8', delimiter=',',
                            converters={"word_vectors": eval})
    test_set = pd.read_csv("Test.csv", usecols=col_list_test, encoding='UTF-8', delimiter=',',
                           converters={"word_vectors": eval})
    label_test = pd.read_csv("Test_labels.csv", usecols=col_list_test_labels, encoding='UTF-8', delimiter=',')

    train_data = train_set.to_numpy()
    test_data = test_set.to_numpy()

    vector_arrays_train = np.empty(len(train_data[:, 1]))
    for i in train_data[:, 1]:
        np.append(vector_arrays_train, np.asarray(i))

    vector_arrays_test = np.empty(len(test_data[:, 1]))
    for i in test_data[:, 1]:
        np.append(vector_arrays_test, np.asarray(i))

    _y_train_ = np.array(train_set['label'])
    _y_test_ = np.array(label_test['label'])
    _x_train = np.column_stack((vector_arrays_train, normalize(np.array(train_data[:, 2:8]))))
    _x_test = np.column_stack((vector_arrays_test, normalize(np.array(test_data[:, 2:]))))
    _y_train = np.expand_dims(_y_train_, 1)
    _y_test = np.expand_dims(_y_test_, 1)
    return _y_train, _y_test, _x_train, _x_test


y_train, y_test, x_train, x_test = load_data()

X = np.concatenate((x_train, x_test))
Y = np.concatenate((y_train, y_test))


# creating model
def model_design():
    model = Sequential()
    model.add(Dense(7, input_dim=7, activation='tanh'))
    model.add(Dense(3, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=model_design, epochs=300, batch_size=100, verbose=2)
kfold = StratifiedKFold(n_splits=5, shuffle=True)
results = cross_val_score(estimator, X[0:10000], Y[0:10000], cv=kfold)
print('Mean Accuracy: ', results.mean())
