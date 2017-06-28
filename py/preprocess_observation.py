# coding: utf-8
from __future__ import unicode_literals
import os
import sys
import fasttext
import numpy as np
import re, math
import itertools
import editdistance
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from hashlib import md5, sha1, sha224
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate import bleu_score

tokenizer = RegexpTokenizer(r'\w+')
w2v_tokenizer = RegexpTokenizer(r'[а-яА-Яa-zA-Z]+')
model = fasttext.load_model('fasttext_models/model_unlemm_ft.bin')

class FeatureExtractor:

    def text_to_vector(self, text):
        words = tokenizer.tokenize(text)
        return Counter(words)

    def cosine(self, reference, candidate):
        vec1 = self.text_to_vector(reference)
        vec2 = self.text_to_vector(candidate)

        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / float(denominator)

    def levenshtein(self, reference, candidate):
        return int(editdistance.eval(reference, candidate))

    def create_trigrams(self, text):
        return zip(*[text[i:] for i in range(3)])

    def jaccard(self, reference, candidate):
        intersection_cardinality = len(set.intersection(*[set(self.create_trigrams(reference)), set(self.create_trigrams(candidate))]))
        union_cardinality = len(set.union(*[set(self.create_trigrams(reference)), set(self.create_trigrams(candidate))]))
        return intersection_cardinality/float(union_cardinality)

    def tfidf_cosine(self, reference, candidate):
        tfidf_vect = TfidfVectorizer()
        refs = [reference, candidate]
        try:
            X = tfidf_vect.fit_transform(refs).toarray()
            row_sums = X.sum(axis=1)
            X = X / row_sums[:, np.newaxis]

            numerator = np.dot(X[0], X[1])

            sum1 = sum(X[0]**2)
            sum2 = sum(X[1]**2)
            denominator = math.sqrt(sum1) * math.sqrt(sum2)

            if not denominator:
                return 0.0
            else:
                return float(numerator) / denominator
        except UnicodeDecodeError:
            return 0

    def bleu(self, reference, candidate):
        try:
            return bleu_score.sentence_bleu([reference], candidate)
        except ZeroDivisionError:
            return 0

    def compute_updates(self, word):
        word = word.encode('utf-8')
        hash1 = lambda x: int(md5(x).hexdigest(), 16)
        hash2 = lambda x: int(sha1(x).hexdigest(), 16)
        hash3 = lambda x: int(sha224(x).hexdigest(), 16)
        dim = 64
        ampl = 10

        pos1 = hash1(word) % dim
        pos2 = hash2(word) % dim
        delta1 = hash3(word) % (2 * ampl + 1) - ampl
        delta2 = ampl - hash3(word) % (2 * ampl + 1)
        return (pos1, delta1), (pos2, delta2)

    def get_updates(self, word):
        updates = dict()
        if word in updates:
            return updates[word]
        else:
            updates[word] = self.compute_updates(word)
            return updates[word]

    def simhash(self, bag_of_words):
        dim = 64
        v = [0] * dim
        wc = Counter(bag_of_words)
        for word, count in wc.items():
            for upd in self.get_updates(word):
                pos, delta = upd
                v[pos] += count * delta

        res = 0
        for i in range(dim):
            if v[i] > 0:
                res |= 1 << i
        return res

    def distance(self, s1, s2):
        dim = 64
        d = 0
        for i in range(dim):
            if s1 & 1 << i != s2 & 1 << i:
                d += 1
        return d

    def simhash_feature(self, reference, candidate):
        s1 = self.simhash(reference.split())
        s2 = self.simhash(candidate.split())
        return self.distance(s1, s2)

    def w2v_vector(self, tokens, num_text, X, vectorizer):
        result_vector = np.zeros(100)
        for token in tokens:
            if sum(model[token]) != 0 and token in vectorizer.get_feature_names():
                index = vectorizer.get_feature_names().index(token)
                tfidf = X[num_text, index]
                weighted = np.multiply(model[token], tfidf)
                result_vector = np.add(result_vector, weighted)
        return result_vector

    def w2v(self, reference, candidate):
        tfidf_vectorizer = TfidfVectorizer()
        X = tfidf_vectorizer.fit_transform([reference, candidate]).toarray()
        tokens1 = [t.lower() for t in w2v_tokenizer.tokenize(reference)]
        tokens2 = [t.lower() for t in w2v_tokenizer.tokenize(candidate)]

        weighted_vec1 = self.w2v_vector(tokens1, 0, X, tfidf_vectorizer)
        weighted_vec2 = self.w2v_vector(tokens2, 1, X, tfidf_vectorizer)

        numerator = np.dot(weighted_vec1, weighted_vec2)

        sum1 = sum(weighted_vec1**2)
        sum2 = sum(weighted_vec2**2)
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator


    def get_features(self, reference, candidate):
        # reference = reference.encode('utf-8')
        # candidate = candidate.encode('utf-8')
        features = [self.cosine(reference, candidate), self.levenshtein(reference, candidate),
                    self.jaccard(reference, candidate), self.tfidf_cosine(reference, candidate),
                    self.bleu(reference, candidate), self.simhash_feature(reference, candidate),
                    self.w2v(reference, candidate)]
        return features
