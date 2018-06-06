
# coding: utf-8

# In[1]:

import numpy as np
from nltk import sent_tokenize
from nltk import word_tokenize
import codecs
import os


# In[2]:

class UnigramModel:
    def __init__(self, freqmodel):
        with codecs.open(freqmodel, 'rU', 'utf-8') as f:
            fq_list=[line.strip("\n").rsplit(',',1) for line in f]
        self.fq_dic={}
        for i in fq_list:
             self.fq_dic[i[0]]=int(i[1])
    def logprob(self, target_word):
        sum_p=float(sum(self.fq_dic.values()))
        if target_word not in self.fq_dic:
            return 0;
        else:
            rp=np.log(float(self.fq_dic[target_word]/sum_p))/np.log(float(2))
            return rp


# In[22]:

def test1(input_dir, output_dir):
    file_list=os.listdir(input_dir);
    for fp in file_list:
        with codecs.open(input_dir+"/"+fp,'rU','utf-8') as f:
            sentences=[sent_tokenize(line) for line in f]
            for i,j in enumerate(sentences):
                for p,q in enumerate(sentences[i]):
                    sentences[i][p]=word_tokenize(sentences[i][p])
            word_freq={}
            for i in sentences:
                for j in i:
                    for k in j:
                        if k not in word_freq:
                            word_freq[k]=1
                        else:
                            word_freq[k]=word_freq[k]+1
        with codecs.open(output_dir+"/"+fp, 'w+', 'utf-8') as f:
            for i,j in word_freq.items():
                f.write(i+","+str(j)+"\n")
        f.close()
        gt_count=get_good_turing(output_dir+"/"+fp)
        with codecs.open("hw2_1_2_"+fp[:-4]+".txt",'w+','utf-8') as f:
            for i,j in gt_count.items():
                f.write(str(i)+"\t"+str(j)+"\n") 
        f.close()


# In[23]:

test1("data/train","freq")


# In[4]:

def get_good_turing(frequency_model):
    with codecs.open(frequency_model, 'rU', 'utf-8') as f:
        fq_list=[line.strip("\n").rsplit(',',1) for line in f]
    num_freq={}
    for i in fq_list:
        if int(i[1]) not in num_freq:
            num_freq[int(i[1])]=1
        else:
            num_freq[int(i[1])]=num_freq[int(i[1])]+1
    r_estimate={}
    for i in range(1,5):
        r_estimate[i]=(i+1)*float(num_freq[i+1])/float(num_freq[i])
    r_estimate[0]=1
    return r_estimate  


# In[5]:

def vocab_size(excerpt):
    with codecs.open(excerpt,'rU','utf-8') as f:
        sentences=[sent_tokenize(line) for line in f]
    word_token=[]
    for i,j in enumerate(sentences):
        for p,q in enumerate(sentences[i]):
            sentences[i][p]=word_tokenize(sentences[i][p])
            for k in sentences[i][p]:
                word_token.append(k)
    return len(set(word_token))


# In[6]:

def frac_freq(excerpt):
    with codecs.open(excerpt,'rU','utf-8') as f:
        sentences=[sent_tokenize(line) for line in f]
    word_token={}
    for i,j in enumerate(sentences):
        for p,q in enumerate(sentences[i]):
            sentences[i][p]=word_tokenize(sentences[i][p])
            for k in sentences[i][p]:
                if k in word_token:
                    word_token[k]=word_token[k]+1
                else:
                    word_token[k]=1
    count=0
    for i in word_token.keys():
        if word_token[i]>5:
            count=count+1
    return float(count)/float(len(word_token.keys()))


# In[7]:

def frac_rare(excerpt):
    with codecs.open(excerpt,'rU','utf-8') as f:
        sentences=[sent_tokenize(line) for line in f]
    word_token={}
    for i,j in enumerate(sentences):
        for p,q in enumerate(sentences[i]):
            sentences[i][p]=word_tokenize(sentences[i][p])
            for k in sentences[i][p]:
                if k in word_token:
                    word_token[k]=word_token[k]+1
                else:
                    word_token[k]=1
    count=0
    for i in word_token.keys():
        if word_token[i]==1:
            count=count+1
    return float(count)/float(len(word_token.keys()))


# In[8]:

def median_word(excerpt):
    with codecs.open(excerpt,'rU','utf-8') as f:
        sentences=[sent_tokenize(line) for line in f]
    word_len=[]
    for i,j in enumerate(sentences):
        for p,q in enumerate(sentences[i]):
            sentences[i][p]=word_tokenize(sentences[i][p])
            for k in sentences[i][p]:
                word_len.append(len(k))
    return np.median(word_len)


# In[9]:

def average_word(excerpt):
    with codecs.open(excerpt,'rU','utf-8') as f:
           sentences=[sent_tokenize(line) for line in f]
    word_len=[]
    for i,j in enumerate(sentences):
        for p,q in enumerate(sentences[i]):
            sentences[i][p]=word_tokenize(sentences[i][p])
            for k in sentences[i][p]:
                word_len.append(len(k))
    return np.mean(word_len)


# In[10]:

def frac_nyt(excerpt):
    with codecs.open(excerpt, 'rU', 'utf-8') as f:
        sentences=[sent_tokenize(line) for line in f]
    word_token=[]
    for i,j in enumerate(sentences):
        for p,q in enumerate(sentences[i]):
            sentences[i][p]=word_tokenize(sentences[i][p])
            for k in sentences[i][p]:
                word_token.append(k)
    word_token=set(word_token)
    with codecs.open("data/train/nytimes.txt",'rU','utf-8') as f:
        sentences=[sent_tokenize(line) for line in f]
    word_in_nytimes=[]
    for i,j in enumerate(sentences):
        for p,q in enumerate(sentences[i]):
            sentences[i][p]=word_tokenize(sentences[i][p])
            for k in sentences[i][p]:
                word_in_nytimes.append(k)
    word_in_nytimes=set(word_in_nytimes)
    word_res=[t for t in word_token if t not in word_in_nytimes]
    return float(len(word_res))/float(len(word_in_nytimes))   


# In[12]:

def test21(input_dir):
    file_list=os.listdir(input_dir)
    with codecs.open("hw2_2_1.txt",'w+','utf-8') as f:
        for fp in file_list:
            vs=vocab_size("data/train/"+fp)
            ff=frac_freq("data/train/"+fp)
            fr=frac_rare("data/train/"+fp)
            mw=median_word("data/train/"+fp)
            aw=average_word("data/train/"+fp)
            fn=frac_nyt("data/train/"+fp)
            f.write(fp[:-4]+","+str(vs)+","+str(ff)+","+str(fr)+","+str(mw)+","+str(aw)+","+str(fn)+"\n")
        f.close()


# In[13]:

def get_type_token_ratio(counts_file):
    with codecs.open(counts_file, 'rU', 'utf-8') as f:
        fq_list=[line.strip("\n").rsplit(',',1) for line in f]
    num=[]
    for i,j in fq_list:
        num.append(int(j))
    return float(len(num))/float(sum(num))


# In[14]:

def get_entropy(unigram_counts_file):
    dicx=UnigramModel(unigram_counts_file)
    for i, j in dicx.fq_dic.items():
        dicx.fq_dic[i]=float(dicx.fq_dic[i])/float(sum(dicx.fq_dic.values()))
    sum_entropy=0
    for i,j in dicx.fq_dic.items():
        sum_entropy=sum_entropy-float(j)*(np.log(float(j))/np.log(2))
    return sum_entropy


# In[30]:

def test22(input_dir):
    file_list=os.listdir(input_dir)
    with codecs.open("hw2_2_2.txt",'w+','utf-8') as f:
        dict_ratio={}
        for i in file_list:
            dict_ratio[i[:-4]]=get_type_token_ratio(input_dir+"/"+i)
        ratio=sorted(dict_ratio.items(), key=lambda d:d[1])
        for i,j in ratio:
            f.write(i+"\n")
        f.close()   


# In[31]:

def test23(input_dir):
    file_list=os.listdir(input_dir)
    entropy_dic={}
    entropy=[]
    for i in file_list:
        entropy_dic[i[:-4]]=get_entropy(input_dir+"/"+i)
        entropy=sorted(entropy_dic.items(),key=lambda d:d[1])
    with codecs.open("hw2_2_3.txt",'w+','utf-8') as f:
        for i,j in entropy:
            f.write(i+"\n")
        f.close()


# In[32]:

test22("freq")


# In[21]:

class BigramModel:
    def __init__(self, trainfiles):
        dic_bigram={}
        unique_list={}
        word_freq={}
        sentences=[]
        for fp in trainfiles:
            with codecs.open(fp,'rU','utf-8') as f:
                sent=[sent_tokenize(line) for line in f]
                sentences=sentences+sent
        for i,j in enumerate(sentences):
            for p,q in enumerate(sentences[i]):
                sentences[i][p]=word_tokenize(sentences[i][p])
                for k in sentences[i][p]:
                    if k not in word_freq:
                        word_freq[k]=1
                    else:
                        word_freq[k]=word_freq[k]+1
        for i,j in word_freq.items():
            if j==1:
                unique_list[i]=1
        for i,j in enumerate(sentences):
            for p,q in enumerate(sentences[i]):
                if len(sentences[i][p])==0:
                    continue
                if sentences[i][p][0] in unique_list:
                    sentences[i][p][0]='<UNK>'
                if '<s>' not in dic_bigram:
                    dic_bigram['<s>']={}
                    dic_bigram['<s>'][sentences[i][p][0]]=1
                else:
                    if sentences[i][p][0] not in dic_bigram['<s>']:
                        dic_bigram['<s>'][sentences[i][p][0]]=1
                    else:
                        dic_bigram['<s>'][sentences[i][p][0]]=dic_bigram['<s>'][sentences[i][p][0]]+1
                for k,l in enumerate(sentences[i][p]):
                    if k==len(sentences[i][p])-1:
                        if l in unique_list:
                            if '<UNK>' not in dic_bigram:
                                dic_bigram['<UNK>']={}
                                dic_bigram['<UNK>']['</s>']=1
                            else:
                                if '</s>' not in dic_bigram['<UNK>']:
                                    dic_bigram['<UNK>']['</s>']=1
                                else:
                                    dic_bigram['<UNK>']['</s>']=dic_bigram['<UNK>']['/s']+1
                        else:
                            if l not in dic_bigram:
                                dic_bigram[l]={}
                                dic_bigram[l]['</s>']=1
                            else:
                                if '</s>' not in dic_bigram[l]:
                                    dic_bigram[l]['</s>']=1
                                else:
                                    dic_bigram[l]['</s>']=dic_bigram[l]['</s>']+1
                    else:
                        if sentences[i][p][k+1] in unique_list:
                            sentences[i][p][k+1]='<UNK>'
                        if l in unique_list:
                            if '<UNK>' not in dic_bigram:
                                dic_bigram['<UNK>']={}
                                dic_bigram['<UNK>'][sentences[i][p][k+1]]=1
                            else:
                                if sentences[i][p][k+1] not in dic_bigram['<UNK>']:
                                    dic_bigram['<UNK>'][sentences[i][p][k+1]]=1
                                else:
                                    dic_bigram['<UNK>'][sentences[i][p][k+1]]=dic_bigram['<UNK>'][sentences[i][p][k+1]]+1
                        else:
                            if l not in dic_bigram:
                                dic_bigram[l]={}
                                dic_bigram[l][sentences[i][p][k+1]]=1
                            else:
                                if sentences[i][p][k+1] not in dic_bigram[l]:
                                    dic_bigram[l][sentences[i][p][k+1]]=1
                                else:
                                    dic_bigram[l][sentences[i][p][k+1]]=dic_bigram[l][sentences[i][p][k+1]]+1
        self.dic_bigram=dic_bigram
    def logprob(self, prior_context, target_word):
        if prior_context in self.dic_bigram:
            if target_word in self.dic_bigram[prior_context]:
                prob=float(self.dic_bigram[prior_context][target_word]+0.25)/float(sum(self.dic_bigram[prior_context].values())+0.25*len(self.dic_bigram.keys()))
            else:
                prob=float(self.dic_bigram[prior_context]['<UNK>']+0.25)/float(sum(self.dic_bigram[prior_context].values())+0.25*len(self.dic_bigram.keys()))
        else:
            if target_word in self.dic_bigram['<UNK>']:
                prob=float(self.dic_bigram['<UNK>'][target_word]+0.25)/float(sum(self.dic_bigram['<UNK>'].values())+0.25*len(self.dic_bigram.keys()))
            else:
                prob=float(self.dic_bigram['<UNK>']['<UNK>']+0.25)/float(sum(self.dic_bigram['<UNK>'].values())+0.25*len(self.dic_bigram.keys()))
        return prob        


# In[1]:

def srilm_preprocess(raw_text, temp_file):
    sent=sent_tokenize(raw_text)
    print(len(sent))
    with codecs.open(temp_file,'w+','utf-8') as f:
        for i in sent:
            f.write(i)
            if i[-1]=='\n':
                continue
            else:
                f.write('\n')
        f.close()


# In[2]:

def srilm_bigram_models(input_file,output_dir):
    os.system("ngram-count -text "+input_file+" -order 1 -addsmooth 0.25 -lm "+output_dir+"/"+input_file[:-4]+".uni.lm")
    os.system("ngram-count -text "+input_file+" -order 2 -addsmooth 0.25 -lm "+output_dir+"/"+input_file[:-4]+".bi.lm")
    os.system("ngram-count -text "+input_file+" -order 2 -kndiscount -lm "+output_dir+"/"+input_file[:-4]+".bi.kn.lm")


# In[33]:

def test4(input_dir):
    file_list=os.listdir(input_dir)
    for i in file_list:
        with codecs.open(input_dir+"/"+i,'rU','utf-8') as f:
            para=[line for line in f]
            para=" ".join(para)
            srilm_preprocess(para,'temp_'+i)


# In[34]:

def srilm_ppl(model_file, raw_text):
    sent=sent_tokenize(raw_text)
    with codecs.open('test.txt','w+','utf-8') as f:
        for i in sent:
            f.write(i)
            if i[-1]=='\n':
                continue
            else:
                f.write('\n')
        f.close()
    return os.system("ngram -lm "+model_file+" -ppl test.txt > perplexity.txt")


# In[ ]:



