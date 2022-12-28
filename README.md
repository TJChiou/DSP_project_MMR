下面的程式建立了一個 MMR with Vector Space Model

參數：句數為原本的 0.2 倍，MMR 公式[^1]中的 λ 設為 0.5，bi-similarity 與 mono-similarity 比例設置為 0.7

並將 text(str) 用此 model 進行摘要，存入 summary。

```python
from mmr import mmr
model = mmr("VSM",ratio=0.2, red_lambda=0.5, monobi_lambda=0.7)
summary = model.to_summary(text) 
```

下面的程式建立了一個 MMR with Sentence Transformer Model

參數：句數為原本的 0.2 倍，MMR 公式[^1]中的 λ 設為 0.5

並將 text(str) 用此 model 進行摘要，存入summary。

```python
from mmr import mmr
model = mmr("DL", ratio=0.2, red_lambda=0.5)
summary = model.to_summary(text)
```

[^1]: **$$ MMR(S_i) = λ × Sim(S_i, D) − (1 − λ) × Sim(S_i, Summ) $$**



