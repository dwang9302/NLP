import numpy as np
import codecs
import os
from nltk import sent_tokenize
from nltk import word_tokenize
from collections import defaultdict

# Helper functions
def build_freq_model(input_path, output_path):
	fq_dict = defaultdict(int)
	with codecs.open(input_path, 'rU', 'utf-8') as f:
		for line in f:
			for sentence in sent_tokenize(line):
				for word in word_tokenize(sentence):
					fq_dict[word] += 1
	to_write_lst = [k+','+str(v) for k, v in fq_dict.items()]
	with codecs.open(output_path,'w+', 'utf-8') as f:
		f.write('\n'.join(to_write_lst))

def get_raw_count(freq_model):
	print freq_model
	raw_count_dict = defaultdict(float)
	with codecs.open(freq_model, 'rU', 'utf-8') as f:
		for line in f:
			word, count = line.rsplit(',', 1)
			raw_count_dict[word] = int(count)
	return raw_count_dict

# 1.1
class UnigramModel:
    def __init__(self, freq_model):
        self.logp_dict = get_raw_count(freq_model)
        sum_fq = float(sum(self.logp_dict.values()))
        for k, v in self.logp_dict.iteritems():
            self.logp_dict[k] = np.log2(v/sum_fq)
            
    def logprob(self, target_word):
            return self.logp_dict[target_word]

# 1.2 
def get_good_turing(freq_model):
	raw_count_dict = get_raw_count(freq_model)
	count_orig = defaultdict(int)
	for _, v in raw_count_dict.iteritems():
		if 0 < v < 7: count_orig[int(v)] += 1
	count_orig[0] = 1
	count_turing = {}
	for i in xrange(6):
		count_turing[i] = (i+1)*count_orig[i+1]/(count_orig[i]*1.0)
	return count_turing

# 1.2 output prep
def get_good_turing_orig(freq_model):
	raw_count_dict = get_raw_count(freq_model)
	count_orig = defaultdict(int)
	for _, v in raw_count_dict.iteritems():
		if 0 < v < 7: count_orig[int(v)] += 1
	count_orig[0] = count_orig[1]
	count_turing = {}
	for i in xrange(6):
		count_turing[i] = (i+1)*count_orig[i+1]/(count_orig[i]*1.0)
	return (count_orig, count_turing)

def output_turing(input_dir):
	files = os.listdir(input_dir)
	for f in files:
		freq_model = input_dir+'/'+f
		count_orig, turing_dict = get_good_turing_orig(freq_model)
		with codecs.open('hw2_1_2_'+f[:-7]+'.txt', 'w+', 'utf-8') as out:
			for i in xrange(6):
				out.write(str(count_orig[i])+'\t'+str(turing_dict[i])+'\n')

# 2.1 
def get_model_stas(input_dir, output_path):
	files = os.listdir(input_dir)
	nyc_file = [f for f in files if f[:2] == 'ny']
	nyc_model = get_raw_count(input_dir+'/'+nyc_file[0])

	for freq_model in files:
		raw_count_dict = get_raw_count(input_dir+'/'+freq_model)
		num_once, num_freq, num_not_nyt = 0.0, 0.0, 0.0
		len_words = []
		for k, v in raw_count_dict.iteritems():
			if v == 1: num_once += 1.0
			if v > 5: num_freq += 1.0
			if k not in nyc_model: num_not_nyt += 1.0
			len_words.append(len(k))
		vocab_size = len(raw_count_dict)
		ret_list = [freq_model[:-7], str(vocab_size), str(num_freq/vocab_size),
				str(num_once/vocab_size), str(np.median(len_words)), 
				str(np.mean(len_words)), str(num_not_nyt/vocab_size)]
		with codecs.open(output_path,'a+', 'utf-8') as f:
			f.write(','.join(ret_list)+'\n')
	
# 2.2 
def get_type_token_ratio()


