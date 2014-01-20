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
DEBUG = True

SAMPLE_TEXT = """In this paper, we present Google, a prototype of a large-scale search engine which makes heavy use of the structure present in hypertext. Google is designed to crawl and index the Web efficiently and produce much more satisfying search results than existing systems. The prototype with a full text and hyperlink database of at least 24 million pages is available at http://google.stanford.edu/ To engineer a search engine is a challenging task. Search engines index tens to hundreds of millions of web pages involving a comparable number of distinct terms. They answer tens of millions of queries every day. Despite the importance of large-scale search engines on the web, very little academic research has been done on them. Furthermore, due to rapid advance in technology and web proliferation, creating a web search engine today is very different from three years ago. This paper provides an in-depth description of our large-scale web search engine -- the first such detailed public description we know of to date. Apart from the problems of scaling traditional search techniques to data of this magnitude, there are new technical challenges involved with using the additional information present in hypertext to produce better search results. This paper addresses this question of how to build a practical large-scale system which can exploit the additional information present in hypertext. Also we look at the problem of how to effectively deal with uncontrolled hypertext collections where anyone can publish anything they want."""

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
        if DEBUG:
          print "Classified text '%s...' as %s" % (text[:25], ','.join(pred))
        return json.dumps({ 'categories': pred }) + "\n"

class HealthResource(resource.Resource):
    def render_GET(self, request):
        health = 'ok' if u'the internet' in get_prediction(SAMPLE_TEXT) else 'unexpected prediction'
        if DEBUG:
          print "Health check: %s" % health
        return json.dumps({ 'health': health }) + "\n"


# Creates NUM_PROCESSES separate Twisted processes listening on the same port
# Adapted from http://stackoverflow.com/questions/10077745/twistedweb-on-multicore-multiprocessor
def main(fd = None):
    root = resource.Resource()
    root.putChild('classify', ClassifyResource())
    root.putChild('health', HealthResource())
    factory = server.Site(root)

    if fd is None:
        port = reactor.listenTCP(PORT, factory)

        if DEBUG:
          print "Started listening on %d" % PORT

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

