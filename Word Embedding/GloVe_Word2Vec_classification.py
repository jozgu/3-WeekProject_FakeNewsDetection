# Code by Alexander Voldby s214591
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from GloVe_and_Word2Vec_preprocessing import corpus, y_corr
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Load in the 2 pretrained word embeddings and save them as dictionaries
twitter_embeddings_index = {}
f = open('twitter_embedding/glove.twitter.27B.100d.txt', 'r', encoding='utf8')
# Create Twitter-embedding dictionary
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    twitter_embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(twitter_embeddings_index))

wikipedia_embeddings_index = {}
f = open('stanford_embedding/glove.6B.100d.txt', 'r', encoding='utf8')
# Create Wikipedia-embedding dictionary
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    wikipedia_embeddings_index[word] = coefs
f.close()

# Load the selftrained word embedding
selftrained = Word2Vec.load('model.selftrained')
vector_length = 100
selftrained_art_matrix = np.zeros((len(corpus), vector_length))

# Article matrix representation using the selftrained word embedding
for i, text in enumerate(corpus):
    vect = [selftrained.wv[w] for w in text]
    x = np.mean(vect, axis=0)
    selftrained_art_matrix[i] = x


# Create matrix representation of the data using each word representation (averaging)

def article_representation(documents, embedding, vector_length):
    """Create vector representation of a corpus of a corpus consisting of articles

    :param documents: A list of documents, where each document must be tokenized
    :param embedding: A dictionary with vector-representations of words
    :param vector_length: The length of the vectors in the chosen embedding
    :return A number of documents x vector_length matrix, where each row is a vector representation of an article
    """
    art_matrix = np.zeros((len(documents), vector_length))

    for i, text in enumerate(documents):
        list = [embedding[w] for w in text if w in embedding.keys()]
        x = np.mean(list, axis=0)
        art_matrix[i] = x

    return art_matrix


wiki_art_matrix = article_representation(corpus, wikipedia_embeddings_index, 100)
twitter_art_matrix = article_representation(corpus, twitter_embeddings_index, 100)

# Run cross validation
wikipedia_score = np.mean(cross_val_score(LogisticRegression(max_iter=1000), wiki_art_matrix, y_corr, cv=5))
twitter_score = np.mean(cross_val_score(LogisticRegression(max_iter=1000), twitter_art_matrix, y_corr, cv=5))
selftrained_score = np.mean(cross_val_score(LogisticRegression(max_iter=1000), selftrained_art_matrix, y_corr, cv=5))

sample_size = np.size(y_corr) // 5

print(f"Wikipedia-trained score: {wikipedia_score}\nTwitter-trained score: {twitter_score}"
      f"\nSelftrained score: {selftrained_score}")

# Confidence intervals
wiki_conf = f"[{wikipedia_score - 1.96*np.sqrt(wikipedia_score*(1-wikipedia_score)/sample_size)};" \
            f" {wikipedia_score + 1.96*np.sqrt(wikipedia_score*(1-wikipedia_score)/sample_size)}]"
twitter_conf = f"[{twitter_score - 1.96*np.sqrt(twitter_score*(1-twitter_score)/sample_size)};" \
            f" {twitter_score + 1.96*np.sqrt(twitter_score*(1-twitter_score)/sample_size)}]"
selftrained_conf = f"[{selftrained_score - 1.96*np.sqrt(selftrained_score*(1-selftrained_score)/sample_size)};" \
            f" {selftrained_score + 1.96*np.sqrt(selftrained_score*(1-selftrained_score)/sample_size)}]"

print(f"Confidence intervals for scores:\n"
      f"Wikipedia-trained: " + wiki_conf + "\nTwitter-trained: " + twitter_conf + "\nSelftrained: " + selftrained_conf)


# Make confusion matrices for each of the models
# Wiki-trained model
X_train, X_test, y_train, y_test = train_test_split(wiki_art_matrix, y_corr, random_state=0)
logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
ConfusionMatrixDisplay.from_estimator(logreg, X_test, y_test, display_labels=["real", "fake"])

# Twitter-trained model
X_train, X_test, y_train, y_test = train_test_split(twitter_art_matrix, y_corr, random_state=0)
logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
ConfusionMatrixDisplay.from_estimator(logreg, X_test, y_test, display_labels=["real", "fake"])

# Selftrained model
X_train, X_test, y_train, y_test = train_test_split(selftrained_art_matrix, y_corr, random_state=0)
logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
ConfusionMatrixDisplay.from_estimator(logreg, X_test, y_test, display_labels=["real", "fake"])

plt.savefig("Confusion_matrices_GloVe_Word2Vec.png")

from Test_on_unseen_data import unseen_embedding_corpus, unseen_embedding_y

# Do the same but use UKDATA as test set.
classifier1 = LogisticRegression(max_iter=1000).fit(
    wiki_art_matrix, y_corr)
score1 = classifier1.score(
    article_representation(unseen_embedding_corpus, wikipedia_embeddings_index, 100), unseen_embedding_y)
ConfusionMatrixDisplay.from_estimator(classifier1, article_representation(
    unseen_embedding_corpus, wikipedia_embeddings_index, 100), unseen_embedding_y, display_labels=["real", "fake"])

classifier2 = LogisticRegression(max_iter=1000).fit(
    twitter_art_matrix, y_corr)
score2 = classifier2.score(
    article_representation(unseen_embedding_corpus, twitter_embeddings_index, 100), unseen_embedding_y)
ConfusionMatrixDisplay.from_estimator(classifier2, article_representation(
    unseen_embedding_corpus, twitter_embeddings_index, 100), unseen_embedding_y, display_labels=["real", "fake"])

unseen_art_matrix = np.zeros((len(unseen_embedding_corpus), 100))
for i, text in enumerate(unseen_embedding_corpus):
    vect = []
    for w in text:
        try:
            vect.append(selftrained.wv[w])
        except KeyError:
            pass
    x = np.mean(vect, axis=0)
    unseen_art_matrix[i] = x

classifier3 = LogisticRegression(max_iter=1000).fit(selftrained_art_matrix, y_corr)
score3 = classifier3.score(
    unseen_art_matrix, unseen_embedding_y)
ConfusionMatrixDisplay.from_estimator(classifier3, unseen_art_matrix, unseen_embedding_y,
                       display_labels=["real", "fake"])

print(f"Wikipedia trained score: {score1}\n Twitter trained score {score2}\n Selftrained score: {score3}")

plt.show()