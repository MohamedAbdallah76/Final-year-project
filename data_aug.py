import random
from random import shuffle
from nltk.corpus import wordnet
import re


class Augment:

    def __init__(self, sentence, num_words, alpha_sr, alpha_ri, p_rd, num_aug):
        self.stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
                           'ours', 'ourselves', 'you', 'your', 'yours',
                           'yourself', 'yourselves', 'he', 'him', 'his',
                           'himself', 'she', 'her', 'hers', 'herself',
                           'it', 'its', 'itself', 'they', 'them', 'their',
                           'theirs', 'themselves', 'what', 'which', 'who',
                           'whom', 'this', 'that', 'these', 'those', 'am',
                           'is', 'are', 'was', 'were', 'be', 'been', 'being',
                           'have', 'has', 'had', 'having', 'do', 'does', 'did',
                           'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
                           'because', 'as', 'until', 'while', 'of', 'at',
                           'by', 'for', 'with', 'about', 'against', 'between',
                           'into', 'through', 'during', 'before', 'after',
                           'above', 'below', 'to', 'from', 'up', 'down', 'in',
                           'out', 'on', 'off', 'over', 'under', 'again',
                           'further', 'then', 'once', 'here', 'there', 'when',
                           'where', 'why', 'how', 'all', 'any', 'both', 'each',
                           'few', 'more', 'most', 'other', 'some', 'such', 'no',
                           'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
                           'very', 's', 't', 'can', 'will', 'just', 'don',
                           'should', 'now', '']

       # words = self.get_only_chars(sentence)
       # n_words = words.split(' ')
       # k_words = [word for word in n_words if word is not '']
       # num_words = len(k_words)

        self.augmented_sentences = []
        num_new_per_technique = int(num_aug / 4) + 1

        # sr
        if alpha_sr > 0:
            n_sr = max(1, int(alpha_sr * num_words))
            for _ in range(num_new_per_technique):
                a_words = self.synonym_replacement(sentence, n_sr)
                self.augmented_sentences.append(' '.join(a_words))

        # ri
        if alpha_ri > 0:
            n_ri = max(1, int(alpha_ri * num_words))
            for _ in range(num_new_per_technique):
                a_words = self.random_insertion(sentence, n_ri)
                self.augmented_sentences.append(' '.join(a_words))


        # rd
        if p_rd > 0:
            for _ in range(num_new_per_technique):
                a_words = self.random_deletion(sentence, p_rd)
                self.augmented_sentences.append(' '.join(a_words))

        self.augmented_sentences = [sentence for sentence in self.augmented_sentences]
        shuffle(self.augmented_sentences)

        # trim so that we have the desired number of augmented sentences
        if num_aug >= 1:
            self.augmented_sentences = self.augmented_sentences[:num_aug]
        else:
            keep_prob = num_aug / len(self.augmented_sentences)
            self.augmented_sentences = [s for s in self.augmented_sentences if random.uniform(0, 1) < keep_prob]

        # append the original sentence
        #self.augmented_sentences.append(sentence)

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(synonym)
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)

    def synonym_replacement(self, words, n):
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in self.stop_words]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if word == random_word else word for word in new_words]
                # print("replaced", random_word, "with", synonym)
                num_replaced += 1
            if num_replaced >= n:  # only replace up to n words
                break


        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')

        return new_words

    def random_deletion(self, words, p):

        #if there's only one word, don't delete it
        if len(words) == 1:
            return words

        # randomly delete words with probability p
        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)

        # if you end up deleting all words, just return a random word
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words) - 1)
            return [words[rand_int]]

        return new_words

    def random_swap(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            new_words = self.swap_word(new_words)
        return new_words

    def swap_word(self, new_words):
        random_idx_1 = random.randint(0, len(new_words) - 1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words) - 1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
        return new_words

    def get_only_chars(self, line):

        clean_line = ""

        line = line.replace("???", "")
        line = line.replace("'", "")
        line = line.replace("-", " ")  # replace hyphens with spaces
        line = line.replace("\t", " ")
        line = line.replace("\n", " ")
        line = line.lower()

        for char in line:
            if char in 'qwertyuiopasdfghjklzxcvbnm ':
                clean_line += char
            else:
                clean_line += ' '

        clean_line = re.sub(' +', ' ', clean_line)  # delete extra spaces
        if clean_line[0] == ' ':
            clean_line = clean_line[1:]
        return clean_line

    def random_insertion(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            self.add_word(new_words)
        return new_words

    def add_word(self, new_words):
        synonyms = []
        counter = 0
        while len(synonyms) < 1:
            random_word = new_words[random.randint(0, len(new_words) - 1)]
            synonyms = self.get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return
        random_synonym = synonyms[0]
        random_idx = random.randint(0, len(new_words) - 1)
        new_words.insert(random_idx, random_synonym)
