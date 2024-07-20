#!/usr/bin/python

from __future__ import print_function

import sys
import os
import json
import numpy as np
from time import time
from random import shuffle

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

# Input directory should have N files [1..N].txt with text documents,
# plus N files [1..N].labels with comma-separated training labels.
input_dir = sys.argv[1]
output_dir = sys.argv[2]

texts = []
labels = []

input_size = len([f for f in os.listdir(input_dir) if f[-3:] == 'txt'])

print('Initial input size: %d' % input_size)
for i in range(input_size):
  try:
    with open("%s/%d.txt" % (input_dir, i)) as txt_file:
      with open("%s/%d.labels" % (input_dir, i)) as lbl_file:
        txt = txt_file.read()
        lbl = json.loads(lbl_file.read())

        texts.append(txt)
        labels.append(lbl)
  except IOError:
    input_size -= 1

print('Final input size: %d' % input_size)

categories = {}
category_list = [item for sublist in labels for item in sublist if (not categories.has_key(item)) and (categories.__setitem__(item, True) or True)]
category_count = len(category_list)
for i in range(category_count):
  categories[category_list[i]] = i

label_vectors = [[(1 if category_list[i] in ls else 0) for i in range(category_count)] for ls in labels]

order = range(input_size)
shuffle(order)

train_texts = [texts[order[i]] for i in range(input_size)]
y_train = np.array([label_vectors[order[i]] for i in range(input_size)])

pipeline_steps = [
    # Count words which are between 3 & 30 characters, appear at least 4 times, appear in less than 50% of documents, and are not known English topic-neutral words.
    ('vectorize', TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=4, stop_words='english', token_pattern='[a-zA-Z]{3,30}')),

    # This should not have an effect at training sample sizes <= 10000 or so, which is fine; it's a safety check to restrict the number of features to something we can train & test on.
    ('reduce_dim', SelectKBest(chi2, k=200000))
]

def train(clf):
    print('=' * 80)
    print("Training: ")
    print(clf)

    steps = pipeline_steps + [('classify', OneVsRestClassifier(clf))]
    pipeline = Pipeline(steps=steps)

    t0 = time()
    pipeline.fit(train_texts, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    return pipeline

# Perceptron & Linear SVM did better than many other models tested which could handle the scale.
models = [
    # For Perceptron, the default alpha did better than alternatives, and "n_iter=50" is blindly taken from an sklearn example.
    # We use manual class weights rather than "auto" in order to raise precision at the expense of recall.
    # (We want to pick "this tag" over "no tag" more often than if we weighted them according to the proportion
    # of actual samples which have any given tag, but less often than if we weighted them equally.)
    Perceptron(class_weight={ 0: 1, 1: 4 }, n_iter=50),

    # For Linear SVM, l2 loss & penalty did better than l1, and the default C=1.0 did better than alternatives.
    # "dual=False" and "tol=1e-3" are blindly taken from an sklearn example.
    # We use auto class weights because manual version tested made recall worse more than they made precision better.
    LinearSVC(loss='l2', penalty='l2', class_weight='auto', dual=False, tol=1e-3)
]

for model in models:
    store = "%s/%s.model" % (output_dir, model.__class__.__name__)
    contents = [train(model), category_list]
    joblib.dump(contents, store)
