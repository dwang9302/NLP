import numpy as np
import codecs
import os
from nltk import sent_tokenize
from nltk import word_tokenize
from collections import defaultdict

class UnigramModel:
    def __init__(self, freqmodel):
        fq_dict = {}
        with codecs.open(freqmodel, 'rU', 'utf-8') as f:
            for line in f:
                w_count = line.rsplit(',', 1)
                fq_dict[w_count[0]] = int(w_count[1])
        sum_fq = float(sum(fq_dict.values()))
        self.logp_dict = defaultdict(float)
        for k, v in self.fq_dict.iteritems():
            self.logp_dict[k] = np.log2(v/sum_fq)
            
    def logprob(self, target_word):
            return self.logp_dict[target_word]
