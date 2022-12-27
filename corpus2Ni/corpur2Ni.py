import json
with open('all_chinese.json', 'r') as f:
    all_chinese = set(json.load(f))

with open('corpus_big.json', 'rb') as f:
    corpus = json.load(f)

mono_word = set()
for data in corpus:
    for w in data['document']:
        if w in all_chinese:
            mono_word.add(w)

mono_Ni = {"":len(corpus)}
# mono_Ni[ empty string ] = N (total number of documents in dataset)

for w in mono_word:
    appear = 0
    for data in corpus:
        if w in data['document']:
            appear += 1
    mono_Ni[w] = appear

with open('mono_Ni.json', 'w') as f:
    json.dump(mono_Ni, f, ensure_ascii=False)


bi_word = set()
bi_word_d = []
for i in range(len(corpus)):
    data = corpus[i]
    bi_word_this_doc = set()
    for i in range( len(data['document'])-1 ):
        if data['document'][i] in all_chinese and data['document'][i+1] in all_chinese:
            bi_word.add(data['document'][i:i+2])
            bi_word_this_doc.add(data['document'][i:i+2])
    bi_word_d.append(bi_word_this_doc)

bi_Ni = {"":len(corpus)}
# bi_Ni[ empty string ] = N (total number of documents in dataset)

for bi_w in bi_word:
    appear = 0
    for i in range(len(corpus)):
        if bi_w in bi_word_d[i]:
            appear += 1
    bi_Ni[bi_w] = appear

with open('bi_Ni.json', 'w') as f:
    json.dump(bi_Ni, f, ensure_ascii=False)