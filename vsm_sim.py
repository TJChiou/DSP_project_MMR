import json
import numpy as np

class vsm_sim:
    def __init__(self,l=0.7):
        with open('mono_Ni.json', 'rb') as f:
            self.mono_Ni = json.load(f)
        with open('bi_Ni.json', 'rb') as f:
            self.bi_Ni = json.load(f)
        self.l=l
    
    def set_up(self,context,doc):
        self.sentences = context
        self.create_doc_mono_embedding(doc)
        self.create_doc_bi_embedding(doc)
        self.create_sentences_mono_embedding_list()
        self.create_sentences_bi_embedding_list()
            
    def create_doc_mono_embedding(self,doc):
        self.mono_vs = set()
        self.mono_cnt = dict()
        for w in doc:
            if w in self.mono_Ni:
                self.mono_vs.add(w)
                if w in self.mono_cnt:
                    self.mono_cnt[w] += 1
                else:
                    self.mono_cnt[w] = 1

        self.mono_vs = list(self.mono_vs)
        self.doc_mono_embedding = [0] * len(self.mono_vs)
        for i in range(len(self.mono_vs)):
            w = self.mono_vs[i]
            TF = 1 + np.log(self.mono_cnt[w])
            IDF = np.log(self.mono_Ni[""]/self.mono_Ni[w])
            self.doc_mono_embedding[i] = TF * IDF
    
    def create_doc_bi_embedding(self,doc):
        self.bi_vs = set()
        self.bi_cnt = dict()
        for i in range(len(doc)-1):
            bi = doc[i:i+2]
            if bi in self.bi_Ni:
                self.bi_vs.add(bi)
                if bi in self.bi_cnt:
                    self.bi_cnt[bi] += 1
                else:
                    self.bi_cnt[bi] = 1

        self.bi_vs = list(self.bi_vs)
        self.doc_bi_embedding = [0] * len(self.bi_vs)
        for i in range(len(self.bi_vs)):
            bi = self.bi_vs[i]
            TF = 1 + np.log(self.bi_cnt[bi])
            IDF = np.log(self.bi_Ni[""]/self.bi_Ni[bi])
            self.doc_bi_embedding[i] = TF * IDF
    
    def create_sentences_mono_embedding_list(self):
        self.sentences_mono_embedding_list = []
        for sen in self.sentences:
            sentences_mono_embedding = [0] * len(self.mono_vs)
            for i in range(len(self.mono_vs)):
                w = self.mono_vs[i]
                if w in sen:
                    sentences_mono_embedding[i] = self.doc_mono_embedding[i]
            self.sentences_mono_embedding_list.append(sentences_mono_embedding)
    
    def create_sentences_bi_embedding_list(self):
        self.sentences_bi_embedding_list = []
        for sen in self.sentences:
            sentences_bi_embedding = [0] * len(self.bi_vs)
            for i in range(len(self.bi_vs)):
                bi = self.bi_vs[i]
                if bi in sen:
                    sentences_bi_embedding[i] = self.doc_bi_embedding[i]
            self.sentences_bi_embedding_list.append(sentences_bi_embedding)

    def c2s_mono_similarity(self,index):
        sen_emb = self.sentences_mono_embedding_list[index]
        doc_emb = self.doc_mono_embedding
        len_sen_emb = sum( [ i*i for i in sen_emb ] )**0.5
        len_doc_emb = sum( [ i*i for i in doc_emb ] )**0.5
        up = sum( [ sen_emb[i] * doc_emb[i] for i in range(len(sen_emb)) ] )
        if len_sen_emb == 0 or len_doc_emb == 0:
            return float(0)
        return up / (len_sen_emb * len_doc_emb)

    def c2s_bi_similarity(self,index):
        sen_emb = self.sentences_bi_embedding_list[index]
        doc_emb = self.doc_bi_embedding
        len_sen_emb = sum( [ i*i for i in sen_emb ] )**0.5
        len_doc_emb = sum( [ i*i for i in doc_emb ] )**0.5
        up = sum( [ sen_emb[i] * doc_emb[i] for i in range(len(sen_emb)) ] )
        if len_sen_emb == 0 or len_doc_emb == 0:
            return float(0)
        return up / (len_sen_emb * len_doc_emb)

    def ss2s_mono_similarity(self,indexs,index):
        sen_emb = self.sentences_mono_embedding_list[index]
        sum_emb = [0] * len(self.mono_vs)
        for i in range(len(self.mono_vs)):
            w = self.mono_vs[i]
            for idx in indexs:
                if w in self.sentences[idx]:
                    sum_emb[i] = self.doc_mono_embedding[i]
                    break
        len_sen_emb = sum( [ i*i for i in sen_emb ] )**0.5
        len_sum_emb = sum( [ i*i for i in sum_emb ] )**0.5
        up = sum( [ sen_emb[i] * sum_emb[i] for i in range(len(sen_emb)) ] )
        if len_sen_emb == 0 or len_sum_emb == 0:
            return float(0)
        return up / (len_sen_emb * len_sum_emb)

    def ss2s_bi_similarity(self,indexs,index):
        sen_emb = self.sentences_bi_embedding_list[index]
        sum_emb = [0] * len(self.bi_vs)
        for i in range(len(self.bi_vs)):
            bi = self.bi_vs[i]
            for idx in indexs:
                if bi in self.sentences[idx]:
                    sum_emb[i] = self.doc_bi_embedding[i]
                    break
        len_sen_emb = sum( [ i*i for i in sen_emb ] )**0.5
        len_sum_emb = sum( [ i*i for i in sum_emb ] )**0.5
        if len_sen_emb == 0 or len_sum_emb == 0:
            return float(0)
        up = sum( [ sen_emb[i] * sum_emb[i] for i in range(len(sen_emb)) ] )
        return up / (len_sen_emb * len_sum_emb)
    
    def c2s_similarity(self,index):
        return self.l * self.c2s_bi_similarity(index) + (1-self.l) * self.c2s_mono_similarity(index)
    
    def ss2s_similarity(self,indexs,index):
        if(len(indexs)==0):
            return float(0)
        return self.l * self.ss2s_bi_similarity(indexs,index) + (1-self.l) * self.ss2s_mono_similarity(indexs,index)

'''
with open('corpus.json', 'rb') as f:
    corpus = json.load(f)
doc = corpus[1]["document"]

v = vsm_sim()
v.split_doc(doc)
v.create_doc_mono_embedding(doc)
v.create_doc_bi_embedding(doc)
v.create_sentences_mono_embedding_list()
v.create_sentences_bi_embedding_list()

print( v.c2s_mono_similarity(3) )
print( v.c2s_bi_similarity(3) )
print( v.ss2s_mono_similarity([1,5,7,9,10,11,12],2) )
print( v.ss2s_bi_similarity([1,5,7,9,10,11,12],2) )
'''

