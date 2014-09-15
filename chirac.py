# -*- coding: utf-8 -*-

import codecs
import re
from tools import *
from gensim import corpora
from gensim import matutils as mtutils
from sklearn import linear_model as lm

fname = "data/chiracLearn"

nblignes = compteLignes(fname)
print "nblignes = %d"%nblignes

alltxts = []
labs = np.ones(nblignes)
s=codecs.open(fname, 'r','utf-8') # pour régler le codage

cpt = 0
for i in range(nblignes):
    txt = s.readline()
    #print txt

    lab = re.sub(r"<[0-9]*:[0-9]*:(.)>.*","\\1",txt)
    txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",txt)

    #assert(lab == "C" or lab == "M")

    if lab.count('M') >0:
        labs[cpt] = -1
    alltxts.append(txt)

    cpt += 1
    if cpt %1000 ==0:
        print cpt
    
stoplist = set('le la les de des à un une en au ne ce d l c s je tu il que qui mais quand'.split())
stoplist.add('')

## DICO
splitters = u'; |, |\*|\. | |\'|'

dictionary = corpora.Dictionary(re.split(splitters, doc.lower()) for doc in alltxts)

print len(dictionary)

stop_ids = [dictionary.token2id[stopword] for stopword in stoplist   if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq < 10 ]
dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
dictionary.compactify() # remove gaps in id sequence after words that were removed

print len(dictionary)

## PROJ

texts = [[word for word in re.split(splitters, document.lower()) if word not in stoplist]  for document in alltxts]
corpus = [dictionary.doc2bow(text) for text in texts]

## exemple de doc
# corpus[0]
# avec les mots
print [dictionary[i] for i,tmp in corpus[0]]

vecteurs = mtutils.corpus2csc(corpus, num_terms=len(dictionary), num_docs=nblignes)
labels = labs

vecteurs = vecteurs.T

classifier = lm.Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, n_iter=5, shuffle=False, verbose=0, eta0=1.0, n_jobs=1, class_weight=None, warm_start=False)
classifier.fit(vecteurs,labels)








