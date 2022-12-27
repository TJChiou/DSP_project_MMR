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

settings=([0.06],[1.0,0.7,0.5,0.3],[1.0,0.7,0.0])
setting_score["VSM"]={}
for setting in itertools.product(*settings):
  model=mmr("VSM",ratio=setting[0],red_lambda=setting[1],monobi_lambda=setting[2])
  print(setting)
  predict_sum=[]
  for data in datas:
    predict_sum.append(' '.join(model.to_summary(data["document"])))
  setting_score["VSM"][setting]=rouge.get_scores(predict_sum, true_sum, avg=True)
  print(setting_score["VSM"][setting])

setting_score["VSM"]={str(key):value for key,value in setting_score["VSM"].items()}
with open("VSM_result.json",'w')as f:
    json.dump(setting_score,f,indent=4)

