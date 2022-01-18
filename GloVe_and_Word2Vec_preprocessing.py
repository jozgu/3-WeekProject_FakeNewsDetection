import numpy as np
import re
import pandas as pd

# Import data
df1 = pd.read_csv("True.csv")
data1 = df1["text"]
df2 = pd.read_csv("Fake.csv")
data2 = df2["text"]
data = pd.concat([data1, data2], ignore_index=True)
y = np.hstack((np.zeros(len(data1)), np.ones(len(data2))))

# Remove Special Characters and append to corpus

dumb_files = []
corpus = []
for i, text in enumerate(data):
    try:
        text_mod = re.sub("[^a-zA-Z]", " ", text)
        text_mod = text_mod.lower()
        text_mod = text_mod.split()

        filt = [w for w in text_mod if len(w) > 1]
        # Also remove the article if it has less than 3 words. Some articles in the dataset are empty
        if len(filt) > 3:
            corpus.append(filt)
        else:
            dumb_files.append(i)

    except TypeError:
        dumb_files.append(i)

y_corr = np.delete(y, dumb_files)
print(len(y_corr))
