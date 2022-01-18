from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases
from GloVe_and_Word2Vec_preprocessing import corpus

# Train the mode
model = Word2Vec(corpus, min_count=0)

# Show vocabulary as  a list
words = list(model.wv.index_to_key)

# Find vectors for specific words
print(model.wv['trump'])
# Find words with high cosine similarity
word_to_test = 'trump'
print(f"Most similar words to {word_to_test} are: {model.wv.most_similar(word_to_test, topn=10)}")

# Save the model
model.save('model.selftrained')
