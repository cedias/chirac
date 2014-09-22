# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 19:16:18 2014

@author: charles
"""

import codecs
import re
from tools import *
import gensim
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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
        labs[i] = -1
    alltxts.append(txt.strip().split(' '))
    cpt += 1
print "loaded %d quotes and labels" % cpt 

alltxtsmul = alltxts*5

model = gensim.models.Word2Vec(alltxtsmul, size=100, window=5, min_count=10, workers=4,sample=1e-5)

reduced_data = PCA(n_components=2).fit_transform(model.syn1)

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

km = KMeans(n_clusters=3, init='k-means++', n_init=3, max_iter=300, tol=0.0001, precompute_distances=False, verbose=0, random_state=None, copy_x=True, n_jobs=1)
km.fit(reduced_data)
