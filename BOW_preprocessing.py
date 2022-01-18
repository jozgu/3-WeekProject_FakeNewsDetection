import numpy as np
import re
from nltk import PorterStemmer
from nltk.corpus import stopwords
from GloVe_and_Word2Vec_preprocessing import data, y

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

failed_files = []
bow_corpus = []
for i, text in enumerate(data):
    if i % 10000 == 0:
        print(i)
    # try-statement benyttes, da nogen artikler blot indeholder nan, hvilket vil give fejl senere
    try:
        text_oa = re.sub("[^a-zA-Z]", " ", text)

        text_oa = text_oa.lower()
        text_oa = text_oa.split()

        # stopwords filtreres fra og artikler med 5 eller færrere ord tilbage fjernes, da vi ikke mener at de bidrager
        # til opfattelsen af fake news. Dette hjælper også til at undgå fejl senere

        filtered_article = [ps.stem(w) for w in text_oa if w not in stop_words and len(w) > 1]
        if len(filtered_article) > 5:
            bow_corpus.append(" ".join(filtered_article))
        else:
            failed_files.append(i)
    except TypeError:
        failed_files.append(i)

# Tilpas target, da nogen artikler er blevet fjernet
print(np.shape(y))
bow_y = np.delete(y, failed_files)
print(np.shape(bow_y))
