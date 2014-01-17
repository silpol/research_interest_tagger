#!/usr/bin/python

# USAGE
#
# python classification_server.py $PATH_TO_MODELS
# curl http://localhost:6606/classify -F text=@$PATH_TO_TEXT_FILE

import os
import sys
import json

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsRestClassifier

from socket import AF_INET
from twisted.internet import reactor
from twisted.web import server, resource

NUM_PROCESSES = 4
PORT = 6606

model_path = sys.argv[1]
perceptron, category_list = joblib.load("%s/Perceptron.model" % model_path, mmap_mode='r')
svm, category_list = joblib.load("%s/LinearSVC.model" % model_path, mmap_mode='r')

def get_model_prediction(model, text):
    pred = model.predict([text])[0]
    return [category_list[i] for i in range(len(pred)) if pred[i] == 1]

def get_prediction(text):
    svm_pred = get_model_prediction(svm, text)
    perc_pred = get_model_prediction(perceptron, text)

    intersect = set(svm_pred) & set(perc_pred)
    if len(intersect) > 0:
      return list(intersect)
    elif len(perc_pred) > 0:
      return perc_pred
    else:
      return svm_pred


class ClassifyResource(resource.Resource):
    def render_POST(self, request):
        text = request.args['text'][0]
        pred = get_prediction(text)
        return json.dumps({ 'categories': pred }) + "\n"


# Creates NUM_PROCESSES separate Twisted processes listening on the same port
# Adapted from http://stackoverflow.com/questions/10077745/twistedweb-on-multicore-multiprocessor
def main(fd = None):
    root = resource.Resource()
    root.putChild('classify', ClassifyResource())
    factory = server.Site(root)

    if fd is None:
        port = reactor.listenTCP(PORT, factory)

        fds = dict([(i,i) for i in range(NUM_PROCESSES-2)])
        fds[port.fileno()] = port.fileno()

        for i in range(NUM_PROCESSES-1):
            reactor.spawnProcess(
                None,
                sys.executable,
                [sys.executable, __file__, model_path, str(port.fileno())],
                childFDs = fds,
                env = os.environ
            )
    else:
        port = reactor.adoptStreamPort(fd, AF_INET, factory)

    reactor.run()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        main()
    else:
        main(int(sys.argv[2]))

