
# coding: utf-8

import os
import pandas as pd
#from __future__ import unicode_literals
import sys
import re
import fileinput
import codecs
import numpy as np
import re
import pickle
import math
import itertools
import editdistance
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate import bleu_score
from nltk.tokenize import RegexpTokenizer
# reload(sys)
# sys.setdefaultencoding('utf-8')

# ## Tools to parse bibliography references

bib_names = [u"литература",
            u"список литературы",
            u"цитируемая литература",
            u"список цитируемой литературы",
            u"список литературных источников",
            u"список использованных источников",
            u"список источников и литературы",
            u"список использованных источников и литературы",
            u"список рекомендуемой литературы",
            u"список использованной литературы",
            u"список публикаций",
            u"список публикаций по теме диссертации",
            u"список работ, опубликованных по теме диссертации",
            u"список основных публикаций по теме диссертации",
            u"список опубликованных работ",
            u"публикации по теме диссертации",
            u"публикации автора по теме диссертации",
            u"работы автора по теме диссертации",
            u"основные публикации по теме диссертации",
            u"библиография",
            u"библиографический список",
            u"ссылки на источники",
            u"список використаних джерел",
            u"збірник наукових праць",
            u"бібліографічний список",
            u"література",
            u"список літератури",
            u"список опублікованих праць за темою дисертації",
            u"статті у наукових фахових виданнях"]


tokenizer = RegexpTokenizer(r'\w+')

def count_occurences_in_text(text, substring_list):
    count_list = []
    for substring in substring_list:
        count_list.append(len([m.start() for m in re.finditer(substring, text.lower())]))
    return count_list


def remove_hyphenation(text):
    # remove empty lines
    text = re.sub("\n\s*\n*", "\n", text)
    dehyphenated = re.sub("^\s|-\n|-\r|\s+$", '', text)
    deh = re.sub("[ ]–\n|[ ]–[ ]\n", " – ", dehyphenated)
    arr = deh.split('\n')
    res = ''
    for a in arr:
        if not a.endswith("."):
            res += a + ' '
        else:
            res += a + '\n'
    return res

def parse_bib_block(text):
    pattern = re.compile("(?:(?<!\d)\d{1,3}(?!\d))\. ?[^\n]* \d{4}")
    num_occurrences = sum(count_occurences_in_text(text, bib_names))
    if num_occurrences == 1:
        count_dict = dict(zip(bib_names, count_occurences_in_text(text, bib_names)))
        for name, val in count_dict.items():
            if val == 1:
                cur_bibname = name

        index = text.lower().index(cur_bibname) + len(cur_bibname)
        bib_block = text[index:].strip()
        strings = bib_block.split("\n")
        for string in strings:
            if pattern.match(string):
                references.append(string)
    elif num_occurrences == 0:
        references = []
    else:
        # Весьма так себе костыль
        references = []
        min_index = 10**7
        for substring in bib_names:
            if substring in text.lower():
                indices = [m.start() for m in re.finditer(substring, text.lower())]
                if indices and min(indices) < min_index:
                    min_index = min(indices)
        strings = text[min_index:].strip().split('\n')
        for string in strings:
            if pattern.match(string):
                references.append(string)
    return references


# ## Collecting bibliography references

data_path = "unlabeled"
paths = os.listdir(data_path)
df = pd.DataFrame(columns=['ref1', 'ref2'])

row = 0
for path in paths:
    dir_name = os.path.join(data_path, path)
    with open(os.path.join(dir_name, 'description.txt')) as text:
        ideal_ref = text.readline()

    print ('ideal', ideal_ref)
    for root, dirs, files in os.walk(dir_name):
        for f in files:
            if not f.endswith('out.txt') and not f.endswith('out-.txt'):
                with open(os.path.join(root, f)) as txt:
                    try:
                        ttt = txt.read()
                        deh = remove_hyphenation(ttt)
                        refs = parse_bib_block(deh)
                        for r in refs:
                            if len(r) > 10:
                                df.loc[row] = [ideal_ref, r]
                                row += 1
                    except UnicodeDecodeError:
                        continue

df['cosine'] = df.apply(cosine, axis=1)
df['levenshtein'] = df.apply(levenshtein, axis=1)
df['jaccard'] = df.apply(jaccard, axis=1)
df['bleu'] = df.apply(bleu, axis=1)
df['tfidf_cosine'] = df.apply(tfidf_cosine, axis=1)
df['simhash'] = df.apply(simhash_feature, axis=1)
df['ft'] = df.apply(weighted_fasttext, axis=1)

df = df.dropna()
df.to_csv('unlabeled.csv', index=False)


# ## Predict labels
with open('rf.pkl', 'rb') as fid:
    rfc = pickle.load(fid)

X = df.drop(['ref1', 'ref2'], axis=1)
df['predicted'] = rfc.predict(X)
df.to_csv('unlabeled_predicted.csv', index=False)


# ## Analyse & count errors
positive = df[df.predicted==1]
negative = df[df.predicted==0]

print ("Analyse positive examples")
for ind, row in positive.iterrows():
    print (ind, row.ref1, '\n', row.ref2, '\n')

print ("Analyse negative examples")
for ind, row in negative.sample(1000).iterrows():
    print (ind, row.ref1, '\n', row.ref2, '\n')
