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

import json
with open("corpus.json",'r') as f:
    corpus = json.load(f)
a = 0
b = 0
for data in corpus:
    a += len(data["summary"])
    b += len(data["document"])
print(a/b)

