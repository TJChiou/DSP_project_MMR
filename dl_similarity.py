from sentence_transformers import SentenceTransformer,util
import numpy as np
import torch
class dl_sim:
    def __init__(self):
        self.model=SentenceTransformer('shibing624/text2vec-base-chinese')

    
    def set_up(self,context,doc):
        self.context_embeddings=self.model.encode(context,convert_to_tensor=True)
    
    def c2s_similarity(self,index):
        return float(util.cos_sim(self.context_embeddings,self.context_embeddings[index]).mean())

    def s2s_similarity(self,index1,index2):
        return float(util.cos_sim(self.context_embeddings[index1],self.context_embeddings[index2]).mean())

    def ss2s_similarity(self,indexs,index):
        if(len(indexs)==0):
            return float(0)
        return float(util.cos_sim(self.context_embeddings[index],torch.stack([self.context_embeddings[i] for i in indexs])).mean())