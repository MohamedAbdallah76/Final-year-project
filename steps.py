import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class Steps:

    def tokenize(data):
        n_data = word_tokenize(data)

        return n_data

    def remove_stop_words(data):
        stop_w = set(stopwords.words('english'))
        words_to_remove = [i for i in stop_w]
        n_data = [i for i in data if i not in words_to_remove]

        return n_data

    def pos_tagging(data):
        n_data = nltk.pos_tag(data)

        return n_data

    # def lemmatizer(data):
    #    lemmatize = WordNetLemmatizer()
    #    n_data = []
    #    words = []
    #    for i in data:
    #        for j in i:
    #            words.append(lemmatize.lemmatize(j))
    #        n_data.append(words)
    #    return n_data
