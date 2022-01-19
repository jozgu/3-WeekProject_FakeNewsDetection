import pandas as pd
import numpy as np
import re
from nltk import PorterStemmer
from nltk.corpus import stopwords
# Load in the small sample of new data:
df = pd.read_csv("Dataset_Unknown Data.csv")
X = df["text"]
y = df["label"]
# Preprocess the data to match the bag of words classifier

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

failed_files = []
unseen_bow_corpus = []
for i, text in enumerate(X):
    # try-statement benyttes, da nogen artikler er tomme, hvilket vil give fejl senere
    text_oa = re.sub("[^a-zA-Z]", " ", text)

    text_oa = text_oa.lower()
    text_oa = text_oa.split()

    filtered_article = [ps.stem(w) for w in text_oa if w not in stop_words and len(w) > 1]
    if len(filtered_article) > 5:
        unseen_bow_corpus.append(" ".join(filtered_article))
    else:
        failed_files.append(i)

# Tilpas target, da nogen artikler er blevet fjernet

unseen_bow_targets = np.delete(y, failed_files)

# Preprocessing, der skal kunne bruges til GloVe og Word2Vec modellerne
dumb_files = []
unseen_embedding_corpus = []
for i, text in enumerate(X):
    try:
        text_mod = re.sub("[^a-zA-Z]", " ", text)
        text_mod = text_mod.lower()
        text_mod = text_mod.split()

        filt = [w for w in text_mod if len(w) > 1]
        if len(filt) > 3:
            unseen_embedding_corpus.append(filt)
        else:
            dumb_files.append(i)

    except TypeError:
        dumb_files.append(i)

unseen_embedding_y = np.delete(y, dumb_files)
