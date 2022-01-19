# Code by Alexander Voldby s214591
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from BOW_preprocessing import bow_corpus, bow_y
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

bow = CountVectorizer().fit(bow_corpus)
bow_vocab_size = len(bow.vocabulary_)
bow_feature_names = bow.get_feature_names()
print(f"Vocabulary size with no reduction: {bow_vocab_size}")
print(f"First 10 features: {bow_feature_names[:10]}\nEvery 1000th feature: {bow_feature_names[::1000]}\n")

# Representation where features occuring in 5 or less documents are removed
reduced_bow = CountVectorizer(min_df=5).fit(bow_corpus)
reduced_bow_vocab_size = len(reduced_bow.vocabulary_)
reduced_bow_feature_names = reduced_bow.get_feature_names()
print(f"Vocabulary size when removing features not present in at least 5 articles: {reduced_bow_vocab_size}")
print(f"First 10 features: {reduced_bow_feature_names[:10]}\nEvery 1000th feature: {reduced_bow_feature_names[::1000]}\n")

# Create matrix and train classifier
X1 = bow.transform(bow_corpus)
classifier1 = LogisticRegression(max_iter=1000).fit(X1, bow_y)

X2 = reduced_bow.transform(bow_corpus)
classifier2 = LogisticRegression(max_iter=1000).fit(X2, bow_y)

# Use a 5-fold cross validation to estimate accuracy
score1 = np.mean(cross_val_score(classifier1, X1, bow_y, cv=5))
score2 = np.mean(cross_val_score(classifier2, X2, bow_y, cv=5))

# Opret en confusion matrix for den ufiltrerede og filtrerede model

# X_train, X_test, y_train, y_test = train_test_split(X1, bow_y, random_state=0)
# print(bow_y)
# logreg = LogisticRegression(max_iter=1000)
# logreg.fit(X_train, y_train)
# ConfusionMatrixDisplay.from_estimator(logreg, X_test, y_test, display_labels=["real", "fake"])

X_train, X_test, y_train, y_test = train_test_split(X2, bow_y, random_state=0)
print(bow_y)
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
ConfusionMatrixDisplay.from_estimator(logreg, X_test, y_test, display_labels=["real", "fake"])

# Brug en Tfidf-vægtet bag of words:
tfidf_vect = TfidfVectorizer(min_df=5)
tfidf_vect.fit(bow_corpus)
tfidf = tfidf_vect.transform(bow_corpus)
classifier3 = LogisticRegression(max_iter=1000).fit(tfidf, bow_y)

X_train, X_test, y_train, y_test = train_test_split(tfidf, bow_y, random_state=0)
logreg_tfidf = LogisticRegression(max_iter=1000)
logreg_tfidf.fit(X_train, y_train)
ConfusionMatrixDisplay.from_estimator(logreg_tfidf, X_test, y_test, display_labels=["real", "fake"])

# Hent de ord, der har den største semantiske vægt
max_value = tfidf.max(axis=0).toarray().ravel()
sorted_by_tfidf = max_value.argsort()
feature_names = np.array(tfidf_vect.get_feature_names())

print("Feature names with lowest tfidf score:\n{}".format(feature_names[sorted_by_tfidf[:10]]))
print("Feature names with highest tfidf score:\n{}".format(feature_names[sorted_by_tfidf[-10:]]))

# Jeg tester modellens generaliseringspotentiale med cross-validation på datasættet.

score3 = np.mean(cross_val_score(LogisticRegression(max_iter=1000), tfidf, bow_y, cv=5))

test_fraction = np.size(bow_y) // 5

print("Unfiltered bag of words average cross-validation score: {:.5f}, confidence interval: [{:.5f}; {:.5f}]".format(
    score1, score1-1.96*np.sqrt(score1*(1-score1)/test_fraction), score1+1.96*np.sqrt(score1*(1-score1)/test_fraction)))
print("Filtered bag of words average cross-validation score: {:.5f}, confidence interval: [{:.5f}; {:.5f}]".format(
    score2, score2-1.96*np.sqrt(score2*(1-score2)/test_fraction), score2+1.96*np.sqrt(score2*(1-score2)/test_fraction)))
print("tfidf average cross-validation score: {:.5f}, confidence interval: [{:.5f}; {:.5f}]".format(
    score3, score3-1.96*np.sqrt(score3*(1-score3)/test_fraction), score3+1.96*np.sqrt(score3*(1-score3)/test_fraction)))

plt.show()
plt.savefig("confusion_matrices_bow.png")

# Test på nyt data
from Test_on_unseen_data import unseen_bow_corpus, unseen_bow_targets

unseen_bow1 = bow.transform(unseen_bow_corpus)

unseen_bow2 = reduced_bow.transform(unseen_bow_corpus)

unseen_bow3 = tfidf_vect.transform(unseen_bow_corpus)

print("Classifier 1 accuracy on unseen dataset: {}".format(
    classifier1.score(unseen_bow1, unseen_bow_targets)))
print("Classifier 2 accuracy on unseen dataset: {}".format(
    classifier2.score(unseen_bow2, unseen_bow_targets)))
print("Classifier 3 accuracy on unseen dataset: {}".format(
    classifier3.score(unseen_bow3, unseen_bow_targets)))

# ConfusionMatrixDisplay.from_estimator(
    # classifier1, unseen_bow1, unseen_bow_targets, display_labels=["real", "fake"])
ConfusionMatrixDisplay.from_estimator(
    classifier2, unseen_bow2, unseen_bow_targets, display_labels=["real", "fake"])
ConfusionMatrixDisplay.from_estimator(
    classifier3, unseen_bow3, unseen_bow_targets,  display_labels=["real", "fake"])
plt.show()