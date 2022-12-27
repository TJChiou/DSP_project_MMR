
import itertools
from mmr import mmr
from rouge_chinese import Rouge

import json
import random

random.seed(66)
with open("corpus.json",'r') as f:
  x=json.load(f)
datas=random.sample(x, 100)
true_sum=[]

for data in datas:
  true_sum.append(' '.join(data["summary"]))
setting_score={}

rouge=Rouge()

settings=([0.02,0.06],[1.0,0.7,0.5,0.3])
setting_score["DL"]={}
for setting in itertools.product(*settings):
  model=mmr("DL",ratio=setting[0],red_lambda=setting[1])
  print(setting)
  predict_sum=[]
  for data in datas:
    predict_sum.append(' '.join(model.to_summary(data["document"])))
  setting_score["DL"][setting]=rouge.get_scores(predict_sum, true_sum, avg=True)
  print(setting_score["DL"][setting])

setting_score["DL"]={str(key):value for key,value in setting_score["DL"].items()}

with open("DL_result.json",'w')as f:
    json.dump(setting_score,f,indent=4)