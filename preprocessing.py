from collections import Counter
import nltk
import pandas as pd
from emoticons import EmoticonDetector
import re as regex
import numpy as np


class TwitterData:
    data = []
    processed_data = []
    wordlist = []
    ngrams = []

    whitelist = ["n't", "not"]
    data_model = None
    data_labels = None
    is_testing = False

    def initialize(self, csv_file, is_testing_set=False, from_cached=None):
        if from_cached is not None:
            self.data_model = pd.read_csv(from_cached)
            return

        self.is_testing = is_testing_set

        if not is_testing_set:
            self.data = pd.read_csv(csv_file, header=0, names=["id", "emotion", "text"])
            self.data = self.data[self.data["emotion"].isin(["positive", "negative", "neutral"])]
        else:
            self.data = pd.read_csv(csv_file, header=0, names=["id", "text"])
            not_null_text = 1 ^ pd.isnull(self.data["text"])
            not_null_id = 1 ^ pd.isnull(self.data["id"])
            self.data = self.data.loc[not_null_id & not_null_text, :]

        self.processed_data = self.data
        self.wordlist = []
        self.data_model = None
        self.data_labels = None

    def cleanup(self, cleanuper):
        t = self.processed_data
        for cleanup_method in cleanuper.iterate():
            if not self.is_testing:
                t = cleanup_method(t)
            else:
                if cleanup_method.__name__ != "remove_na":
                    t = cleanup_method(t)

        self.processed_data = t

    def build_features(self):
        def count_by_lambda(expression, word_array):
            return len(list(filter(expression, word_array)))

        def count_occurences(character, word_array):
            counter = 0
            for j, word in enumerate(word_array):
                for char in word:
                    if char == character:
                        counter += 1

            return counter

        def count_by_regex(regex, plain_text):
            return len(regex.findall(plain_text))

        self.add_column("splitted_text", map(lambda txt: txt.split(" "), self.processed_data["text"]))

        # number of uppercase words
        uppercase = list(map(lambda txt: count_by_lambda(lambda word: word == word.upper(), txt),
                             self.processed_data["splitted_text"]))
        self.add_column("number_of_uppercase", uppercase)

        # number of !
        exclamations = list(map(lambda txt: count_occurences("!", txt),
                                self.processed_data["splitted_text"]))

        self.add_column("number_of_exclamation", exclamations)

        # number of ?
        questions = list(map(lambda txt: count_occurences("?", txt),
                             self.processed_data["splitted_text"]))

        self.add_column("number_of_question", questions)

        # number of ...
        ellipsis = list(map(lambda txt: count_by_regex(regex.compile(r"\.\s?\.\s?\."), txt),
                            self.processed_data["text"]))

        self.add_column("number_of_ellipsis", ellipsis)

        # number of hashtags
        hashtags = list(map(lambda txt: count_occurences("#", txt),
                            self.processed_data["splitted_text"]))

        self.add_column("number_of_hashtags", hashtags)

        # number of mentions
        mentions = list(map(lambda txt: count_occurences("@", txt),
                            self.processed_data["splitted_text"]))

        self.add_column("number_of_mentions", mentions)

        # number of quotes
        quotes = list(map(lambda plain_text: int(count_occurences("'", [plain_text.strip("'").strip('"')]) / 2 +
                                                 count_occurences('"', [plain_text.strip("'").strip('"')]) / 2),
                          self.processed_data["text"]))

        self.add_column("number_of_quotes", quotes)

        # number of urls
        urls = list(map(lambda txt: count_by_regex(regex.compile(r"http.?://[^\s]+[\s]?"), txt),
                        self.processed_data["text"]))

        self.add_column("number_of_urls", urls)

        # number of positive emoticons
        ed = EmoticonDetector()
        positive_emo = list(
            map(lambda txt: count_by_lambda(lambda word: ed.is_emoticon(word) and ed.is_positive(word), txt),
                self.processed_data["splitted_text"]))

        self.add_column("number_of_positive_emo", positive_emo)

        # number of negative emoticons
        negative_emo = list(map(
            lambda txt: count_by_lambda(lambda word: ed.is_emoticon(word) and not ed.is_positive(word), txt),
            self.processed_data["splitted_text"]))

        self.add_column("number_of_negative_emo", negative_emo)

        pass

    def add_column(self, column_name, column_content):
        self.processed_data.loc[:, column_name] = pd.Series(column_content, index=self.processed_data.index)

    def stem(self, stemmer=nltk.PorterStemmer()):
        def stem_and_join(row):
            row["text"] = list(map(lambda str: stemmer.stem(str.lower()), row["text"]))
            return row

        self.processed_data = self.processed_data.apply(stem_and_join, axis=1)

    def tokenize(self, tokenizer=nltk.word_tokenize):
        def tokenize_row(row):
            row["text"] = tokenizer(row["text"])
            row["tokenized_text"] = [] + row["text"]
            return row

        self.processed_data = self.processed_data.apply(tokenize_row, axis=1)

    def build_wordlist(self, min_occurrences=5, max_occurences=500, stopwords=nltk.corpus.stopwords.words("english"),
                       whitelist=None):
        self.wordlist = []
        whitelist = self.whitelist if whitelist is None else whitelist
        import os
        if os.path.isfile("data\\wordlist.csv"):
            word_df = pd.read_csv("data\\wordlist.csv")
            word_df = word_df[word_df["occurrences"] > min_occurrences]
            self.wordlist = list(word_df.loc[:, "word"])
            return

        words = Counter()
        for idx in self.processed_data.index:
            words.update(self.processed_data.loc[idx, "text"])

        for idx, stop_word in enumerate(stopwords):
            if stop_word not in whitelist:
                del words[stop_word]

        word_df = pd.DataFrame(data={"word": [k for k, v in words.most_common() if min_occurrences < v < max_occurences],
                                     "occurrences": [v for k, v in words.most_common() if min_occurrences < v < max_occurences]},
                               columns=["word", "occurrences"])

        word_df.to_csv("data\\wordlist.csv", index_label="idx")
        self.wordlist = [k for k, v in words.most_common() if min_occurrences < v < max_occurences]

    def build_ngrams(self, ngram=2, stopwords=nltk.corpus.stopwords.words("english"),
                     whitelist=None):
        whitelist = self.whitelist if whitelist is None else whitelist
        stopwords = list(filter(lambda sw: sw not in whitelist, stopwords))
        ngrams = Counter()
        for idx in self.processed_data.index:
            tokens = self.processed_data.loc[idx, "text"]
            ngrams.update(self.generate_ngrams(tokens, ngram, stopwords))

        self.ngrams = [ng for ng, cnt in ngrams.most_common() if cnt >= 2]

    def generate_ngrams(self, tokens, ngram, stopwords):
        return list(map(lambda ng: str.join("_", ng),
                        nltk.ngrams(
                            filter(lambda word: word not in stopwords, tokens),
                            ngram)))

    def build_ngram_model(self, stopwords=nltk.corpus.stopwords.words("english"),
                          whitelist=None, ngram=2):
        whitelist = self.whitelist if whitelist is None else whitelist
        stopwords = list(filter(lambda sw: sw not in whitelist, stopwords))
        extra_columns = [col for col in self.processed_data.columns if col.startswith("number_of")]
        label_column = []
        if not self.is_testing:
            label_column = ["label"]

        columns = label_column + extra_columns + list(self.ngrams)
        labels = []
        rows = []
        for idx in self.processed_data.index:
            current_row = []

            if not self.is_testing:
                # add label
                current_label = self.processed_data.loc[idx, "emotion"]
                labels.append(current_label)
                current_row.append(current_label)

            for _, col in enumerate(extra_columns):
                current_row.append(self.processed_data.loc[idx, col])

            # add ngrams
            tokens = self.processed_data.loc[idx, "text"]
            current_ngrams = self.generate_ngrams(tokens, ngram, stopwords)
            for _, ng in enumerate(self.ngrams):
                current_row.append(1 if ng in current_ngrams else 0)

            rows.append(current_row)

        self.data_model = pd.DataFrame(rows, columns=columns)
        self.data_labels = pd.Series(labels)
        return self.data_model, self.data_labels

    def build_word2vec_model(self, word2vec_provider, stopwords=nltk.corpus.stopwords.words("english"), whitelist=None):
        whitelist = self.whitelist if whitelist is None else whitelist
        stopwords = list(filter(lambda sw: sw not in whitelist, stopwords))
        extra_columns = [col for col in self.processed_data.columns if col.startswith("number_of")]
        similarity_columns = ["bad_similarity", "good_similarity", "information_similarity"]
        label_column = []
        if not self.is_testing:
            label_column = ["label"]

        columns = label_column + ["original_id"] + extra_columns + similarity_columns + list(
            map(lambda i: "word2vec_{0}".format(i), range(0, word2vec_provider.dimensions))) + list(
            map(lambda w: w + "_bow",self.wordlist))
        labels = []
        rows = []
        for idx in self.processed_data.index:
            current_row = []

            if not self.is_testing:
                # add label
                current_label = self.processed_data.loc[idx, "emotion"]
                labels.append(current_label)
                current_row.append(current_label)

            current_row.append(self.processed_data.loc[idx, "id"])

            for _, col in enumerate(extra_columns):
                current_row.append(self.processed_data.loc[idx, col])

            # average similarities with words
            tokens = self.processed_data.loc[idx, "tokenized_text"]
            for main_word in map(lambda w: w.split("_")[0], similarity_columns):
                current_similarities = [abs(sim) for sim in
                                        map(lambda word: word2vec_provider.get_similarity(main_word, word.lower()), tokens) if
                                        sim is not None]
                if len(current_similarities) <= 1:
                    current_row.append(0 if len(current_similarities) == 0 else current_similarities[0])
                    continue
                max_sim = max(current_similarities)
                min_sim = min(current_similarities)
                current_similarities = [((sim - min_sim) / (max_sim - min_sim)) for sim in
                                        current_similarities]  # normalize to <0;1>
                current_row.append(np.array(current_similarities).mean())

            # add word2vec vector
            tokens = self.processed_data.loc[idx, "tokenized_text"]
            current_word2vec = []
            for _, word in enumerate(tokens):
                vec = word2vec_provider.get_vector(word.lower())
                if vec is not None:
                    current_word2vec.append(vec)

            averaged_word2vec = list(np.array(current_word2vec).mean(axis=0))
            # averaged_word2vec = map(lambda avg: (avg if abs(avg) > 0.0001 else 0), averaged_word2vec)
            current_row += averaged_word2vec

            # add bag-of-words
            tokens = set(self.processed_data.loc[idx, "text"])
            for _, word in enumerate(self.wordlist):
                current_row.append(1 if word in tokens else 0)

            rows.append(current_row)

        self.data_model = pd.DataFrame(rows, columns=columns)
        self.data_labels = pd.Series(labels)
        return self.data_model, self.data_labels

    def build_data_model(self, with_ngram=None):
        extra_columns = [col for col in self.processed_data.columns if col.startswith("number_of")]
        label_column = []
        if not self.is_testing:
            label_column = ["label"]

        columns = label_column + extra_columns + list(
            map(lambda w: w + "_bow",self.wordlist))
        if with_ngram is not None:
            columns += list(self.ngrams)
        labels = []
        rows = []
        for idx in self.processed_data.index:
            current_row = []

            if not self.is_testing:
                # add label
                current_label = self.processed_data.loc[idx, "emotion"]
                labels.append(current_label)
                current_row.append(current_label)

            for _, col in enumerate(extra_columns):
                current_row.append(self.processed_data.loc[idx, col])

            # add tokens
            tokens = set(self.processed_data.loc[idx, "text"])
            for _, word in enumerate(self.wordlist):
                current_row.append(1 if word in tokens else 0)

            if with_ngram is not None:
                current_ngrams = self.generate_ngrams(self.processed_data.loc[idx, "text"], with_ngram, [])
                for _, ng in enumerate(self.ngrams):
                    current_row.append(1 if ng in current_ngrams else 0)

            rows.append(current_row)

        self.data_model = pd.DataFrame(rows, columns=columns)
        self.data_labels = pd.Series(labels)
        return self.data_model, self.data_labels
