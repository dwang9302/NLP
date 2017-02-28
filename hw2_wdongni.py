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
			len_words.extend([len(k)]*v)
		vocab_size = len(raw_count_dict)
		ret_list = [freq_model[:-7], str(vocab_size), str(num_freq/vocab_size),
				str(num_once/vocab_size), str(np.median(len_words)), 
				str(np.mean(len_words)), str(num_not_nyt/vocab_size)]
		with codecs.open(output_path,'a+', 'utf-8') as f:
			f.write(','.join(ret_list)+'\n')
	
# 2.2 
def get_type_token_ratio(freq_model):
	raw_count_dict = get_raw_count(freq_model)
	return len(raw_count_dict)*1.0/sum(raw_count_dict.values())

# 2.2 output prep
def output_tt_ratio(input_dir):
	files = os.listdir(input_dir)
	res = [(get_type_token_ratio(input_dir+'/'+f), f[:-7])for f in files]
	ret = [fname for r, fname in sorted(res)]
	with codecs.open('hw2_2_2.txt', 'w+', 'utf-8') as f:
		f.write('\n'.join(ret))

# 2.3
def get_entropy(freq_model):
	raw_count_dict = get_raw_count(freq_model)
	tokens = sum(raw_count_dict.values())
	prop_lst = []
	for v in raw_count_dict.values():
		prop_lst.append(float(v)/tokens)
	return sum([-v*np.log2(v) for v in prop_lst])

def output_entropy(input_dir):
	files = os.listdir(input_dir)
	res = [(get_entropy(input_dir+'/'+f), f[:-7])for f in files]
	ret = [fname for r, fname in sorted(res)]
	print res
	with codecs.open('hw2_2_3.txt', 'w+', 'utf-8') as f:
		f.write('\n'.join(ret))

# 3.1 UNK, 0.25 smoothing
# t_files = ['data/train/cancer.txt', 'data/train/nytimes.txt', 'data/train/obesity.txt']
class BigramModel:
	def __init__(self, trainfiles):
		# get fq_dict with <s> and </s> 
		fq_dict = defaultdict(int)
		self.rare_word = '<UNK>'
		self.smoothing = 0.25
		all_files = []
		self.bigram_dict = defaultdict(lambda: defaultdict(int))
		for input_f in trainfiles:
			file_tokens = []
			print input_f
			with codecs.open(input_f, 'rU', 'utf-8') as f:
				for line in f:
					for sentence in sent_tokenize(line):
						file_tokens.append('<s>')
						the_words = word_tokenize(sentence)
						for w in the_words:
							fq_dict[w] += 1
						file_tokens.extend(the_words)
						file_tokens.append('</s>')
			all_files.append(file_tokens)
			
			print 'finished counting ', input_f

		rare_set = set()
		for k, v in fq_dict.iteritems():
			if v == 1:
				rare_set.add(k)

		# get the bigram_dict (take care of rare words)
		for file_tokens in all_files:
			for i in xrange(1, len(file_tokens)):
				last_word, curr_word = file_tokens[i-1], file_tokens[i]
				if last_word in rare_set: last_word = self.rare_word				
				if curr_word in rare_set: curr_word = self.rare_word
				self.bigram_dict[last_word][curr_word] += 1
		# build the logprob dictionary
		self.v_size = len(fq_dict)-len(rare_set)
		self.logp_dict = defaultdict(dict)
		for prior in self.bigram_dict.keys():
			condi_sum = sum(self.bigram_dict[prior].values())+self.smoothing*self.v_size
			for target in self.bigram_dict[prior].keys():
				self.logp_dict[prior][target] = np.log2((self.bigram_dict[prior][target]*1.0
														+self.smoothing)/condi_sum)
			if self.rare_word not in self.bigram_dict[prior]:
				self.logp_dict[prior][self.rare_word] = np.log2(1.0/condi_sum)


	def logprob(self, prior_context, target_word):
		if prior_context not in self.logp_dict:
			prior_context = self.rare_word
		if prior_context not in self.logp_dict: # <UNK> not in corpus
			return np.log2(1.0/self.v_size)
		if target_word not in self.logp_dict[prior_context]:
			target_word = self.rare_word
		return self.logp_dict[prior_context][target_word]


# 4.1

def srilm_preprocess(raw_text, temp_file):
	res = []
	with codecs.open(raw_text, 'rU', 'utf-8') as f:
		for line in f:
			res.extend(sent_tokenize(line))
	with codecs.open(temp_file, 'w+', 'utf-8') as f:
		f.write('\n'.join(res))


def srilm_bigram_models(input_file,output_dir):
	base_name = os.path.basename(input_file)
	processed = 'temp_'+base_name
	srilm_preprocess(input_file, processed)
	outputs = [output_dir+"/"+base_name+".uni.lm", output_dir+"/"+base_name+".bi.lm", output_dir+"/"+base_name+".bi.kn.lm"]
	for f in outputs:
		codecs.open(f, 'w+').close()
	os.system("/home1/c/cis530/srilm/ngram-count -text "+processed+" -order 1 -addsmooth 0.25 -lm "+outputs[0])
	os.system("/home1/c/cis530/srilm/ngram-count -text "+processed+" -order 2 -addsmooth 0.25 -lm "+outputs[1])
	os.system("/home1/c/cis530/srilm/ngram-count -text "+processed+" -order 2 -kndiscount -lm "+outputs[2])

# 4.2

def srilm_ppl(model_file, raw_text):
	sent = sent_tokenize(raw_text)
	with codecs.open('test.txt','w+','utf-8') as f:
		f.write('\n'.join(sent))
	return os.system("/home1/c/cis530/srilm/ngram -lm "+model_file+" -ppl test.txt > perplexity.txt")
						


