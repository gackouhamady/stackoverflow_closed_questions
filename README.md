# Predict Closed Questions on Stack Overflow (Kaggle, 2012)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-%E2%89%A5%201.5-150458)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.x-F89939)
![NLP](https://img.shields.io/badge/Domain-NLP-8A2BE2)
![Vectorizer](https://img.shields.io/badge/Features-HashingVectorizer-informational)
![Classifier](https://img.shields.io/badge/Model-SGDClassifier-success)
![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF)
![GPU](https://img.shields.io/badge/Accelerator-GPU%20P100-2ea44f)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> Multiclass text classification to predict whether a Stack Overflow question will be **closed**, and **for which reason**.  
> Official classes: `not a real question`, `not constructive`, `off topic`, `open`, `too localized`.  
> Metric: **Multiclass Log Loss** (submit probabilities).

---

## 📂 Data
- Training covers data **through July 31 (UTC)**.  
- Public leaderboard data covers **Aug 1–14 (UTC)**.  
- Key files:
  - `train.csv` — text (Title, BodyMarkdown), tags (Tag1–Tag5), user metadata, target `OpenStatus`.
  - `public_leaderboard.csv` — same inputs, **without** `OpenStatus`.
  - `train-sample.csv` — stratified sample (all closed + equal-sized sample of open).
- On Kaggle Notebooks, datasets are auto-mounted under:
/kaggle/input/predict-closed-questions-on-stack-overflow/

> The competition is archived (submissions disabled), but this repo reproduces the challenge locally and via Kaggle Notebooks.

---

## ⚙️ Tech Stack
- **Language**: Python 3.11  
- **Libraries**: pandas, scikit-learn  
- **Features**: `HashingVectorizer` (word + bigrams)  
- **Model**: `SGDClassifier(loss="log_loss")` (fast, probabilistic)  
- **Runtime Env**: Kaggle Notebook (GPU P100 enabled; not required for linear models)

---

## 🚀 Quick Start (Kaggle Notebook)

1. Open a new Kaggle Notebook on the competition page → **Code → New Notebook**.  
2. Paste the minimal cell below and run it to produce `submission.csv`:

```python
# Fast prototype: hashing + SGD (probabilities)
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

train = pd.read_csv("/kaggle/input/predict-closed-questions-on-stack-overflow/train.csv")
test  = pd.read_csv("/kaggle/input/predict-closed-questions-on-stack-overflow/public_leaderboard.csv")

classes = ["not a real question","not constructive","off topic","open","too localized"]

# quick stratified downsample for speed
per_class = 4000  # lower (e.g., 2000) to run even faster
parts = [train[train["OpenStatus"] == c].sample(n=min(per_class, (train["OpenStatus"] == c).sum()), random_state=42)
       for c in classes]
train_small = pd.concat(parts, ignore_index=True)

Xtr_text = (train_small["Title"].fillna("") + " " + train_small["BodyMarkdown"].fillna(""))
Xte_text = (test["Title"].fillna("") + " " + test["BodyMarkdown"].fillna(""))

hv = HashingVectorizer(n_features=2**18, ngram_range=(1,2), alternate_sign=False)
Xtr, Xte = hv.transform(Xtr_text), hv.transform(Xte_text)

clf = SGDClassifier(loss="log_loss", max_iter=5, tol=1e-3, n_jobs=-1, random_state=42)
clf.fit(Xtr, train_small["OpenStatus"])

def softmax(z):
  z = z - z.max(axis=1, keepdims=True); ez = np.exp(z); return ez / ez.sum(axis=1, keepdims=True)

proba = getattr(clf, "predict_proba", None)
proba_all = clf.predict_proba(Xte) if proba else softmax(clf.decision_function(Xte))

sub = pd.DataFrame(proba_all, columns=clf.classes_).reindex(columns=classes, fill_value=0)
if "id" in test.columns: sub.insert(0, "id", test["id"].values)
sub.to_csv("submission.csv", index=False)
print("submission.csv written (fast prototype).")
```
Since the official submissions are closed, evaluate locally (log loss) or keep submission.csv for your internal leaderboard.
