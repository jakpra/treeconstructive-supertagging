'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

import sys
import bcolz
import pickle
import numpy as np

glove_path = sys.argv[1]  # 'C:\\Users\\Jakob\\Downloads\\glove.840B.300d'
size = int(sys.argv[2])  # 840
dim = int(sys.argv[3])  # 300

words = []
idx = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/{size}B.{dim}.dat', mode='w')

with open(f'{glove_path}/glove.{size}B.{dim}d.txt', 'rb') as f:
    for l in f:
        line = l.split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)

vectors = bcolz.carray(vectors[1:].reshape((-1, dim)), rootdir=f'{glove_path}/{size}B.{dim}.dat', mode='w')
vectors.flush()
pickle.dump(words, open(f'{glove_path}/{size}B.{dim}_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'{glove_path}/{size}B.{dim}_idx.pkl', 'wb'))
