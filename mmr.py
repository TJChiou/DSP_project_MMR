import vsm_sim
import dl_similarity
import math

class mmr:

    def split_context(doc):
        sentences = []
        delimiters = ' ；，。！？'
        s = ''
        for w in doc:
            if w in delimiters:
                if s == '':
                    continue
                else:
                    sentences.append(s+w)
                    s = ''
            else:
                s += w
        if s != '':
            sentences.append(s)
        return sentences
    
    def __init__(self,config,ratio=0.25,red_lambda=0.7,monobi_lambda=0.7):
        self.l=red_lambda
        if(config=="VSM"):
            self.model=vsm_sim.vsm_sim(l=monobi_lambda)
        else:
            self.model=dl_similarity.dl_sim()
        self.ratio=ratio
    
    def to_summary(self,doc):
        sentences=mmr.split_context(doc)
        self.model.set_up(sentences,doc)
        output_length=math.ceil(self.ratio*len(sentences))
        output=set()
        Rel=[]
        for i in range(len(sentences)):
            Rel.append(self.model.c2s_similarity(i))

        for epoch in range(output_length):
            Red=[]
            mmr_score=[]
            for i in range(len(sentences)):
                if(i in output):
                    Red.append(0)
                    mmr_score.append(-math.inf)
                    continue
                Red.append(self.model.ss2s_similarity(output,i))
                mmr_score.append(self.l*Rel[i]-(1-self.l)*Red[i])
            output.add(mmr_score.index(max(mmr_score)))
        return ''.join([sentences[i] for i in sorted(list(output))])

