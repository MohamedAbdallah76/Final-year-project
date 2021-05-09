from collections import Counter
import textstat
import vaderSentiment.vaderSentiment as vd
import numpy


class Features:

    def formality(data, number):
        count_articles = 0
        num_words = number
        for i in data:
            if i[0] == "the" or i[0] == "a" or i[0] == "an":
                count_articles += 1

        counts = Counter(data for word, data in data)
        if num_words == 0:
            noun_freq = 0
            adj_freq = 0
            prep_freq = 0
            article_freq = 0
            pronoun_freq = 0
            verb_freq = 0
            adverb_freq = 0
            inter_freq = 0
        else:
            noun_freq = ((counts["NN"] + counts["NNP"] + counts["NNS"]) / num_words) * 100
            adj_freq = (counts["JJS"] / num_words) * 100
            prep_freq = (counts["IN"] / num_words) * 100
            article_freq = (count_articles / num_words) * 100
            pronoun_freq = ((counts["PRP"] + counts["PRP$"]) / num_words) * 100
            verb_freq = ((counts["VB"] + counts["VBD"] + counts["VBG"] + counts["VBP"] +
                          counts["VBZ"]) / num_words) * 100
            adverb_freq = ((counts["RBR"] + counts["RBS"] + counts["RB"]) / num_words) * 100
            inter_freq = (counts["UH"] / num_words) * 100

        form_m = (noun_freq + adj_freq + prep_freq + article_freq - pronoun_freq - verb_freq -
                  adverb_freq - inter_freq + 100) / 2

        return form_m

    def flesch_re(data):

        FRE = textstat.textstat.flesch_reading_ease(data)

        return FRE

    def avg_word_per_sentence(data):

        avg_WPS = textstat.textstat.avg_sentence_length(data)

        return avg_WPS

    def avg_syllables_PW(data):

        avg_SPW = textstat.textstat.avg_syllables_per_word(data, interval=None)

        return avg_SPW

    def num_difficult_words(data):

        count_DF = textstat.textstat.difficult_words(data, syllable_threshold=2)

        return count_DF

    def sentiment(data):
        sentiment = []
        analyzer = vd.SentimentIntensityAnalyzer()
        for i in data:
            result = analyzer.polarity_scores(i)["compound"]
            sentiment.append(result)
        if len(sentiment) == 0:
            return 0
        else:
            return sum(sentiment) / len(sentiment)
