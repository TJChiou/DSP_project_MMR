import datasets
import json
import re

dataset=datasets.load_dataset('wiki_lingua','chinese',split='train').shuffle(seed=11)

document_summary=[]
count=0

for item in dataset["article"]:
    for i in range(len(item['document'])):
        item['document'][i]=re.sub('{.*?}', '', item['document'][i])
        item['summary'][i]=re.sub('{.*?}', '', item['summary'][i])
        if(len(item['document'][i])>1000):
            document_summary.append(
                {
                    'document':item['document'][i],
                    'summary':item['summary'][i]
                }
            )
            count+=1
            
with open("corpus.json",'w',encoding='utf8') as f:
    json.dump(document_summary,f,ensure_ascii=False,indent=4)