# Code by Alexander Voldby s214591
import mglearn
import numpy as np
import matplotlib.pyplot as plt
from BOW_preprocessing import bow_corpus, bow_y
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# The code for making this model is based on code from "Introdution to machine learning with python"
# by Andreas Müller and Sarah Guido

pipe = make_pipeline(CountVectorizer(min_df=5),
                    LogisticRegression(max_iter=1000))
param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]}

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(bow_corpus, bow_y)

vectorizer = grid.best_estimator_.named_steps["countvectorizer"]
feature_names = np.array(vectorizer.get_feature_names())

fig = plt.figure()
mglearn.tools.visualize_coefficients(grid.best_estimator_.named_steps["logisticregression"].coef_,
                                     feature_names, n_top_features=40)

plt.show()
fig.savefig('words_in_fake_news_detection.png')