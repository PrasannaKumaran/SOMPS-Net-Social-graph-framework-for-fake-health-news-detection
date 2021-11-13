import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from numpy import save
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tqdm.pandas()

def basic_cleanup(data):
    '''
    Preprocess a given text
    Parameters:
        data (String) : input text
    Returns:
        data (String) : output text
    '''
    data = data.lower()
    data = re.sub(r'^RT[\s]+', '', data)
    data = re.sub(r'https?:\/\/.*[\r\n]*', '', data)
    data = re.sub(r'#', '', data)
    data = re.sub(r'[0-9]', '', data)
    data = re.sub(r'@[a-zA-Z0-9_]*', '', data)
    data = re.sub(r':', '', data)
    data = re.sub(r'[,.\'“-]', '', data)
    return data

def apply_embedding(x):
    '''
    Apply word embeddings for a given text
    Parameters:
        x (String): input text
    Returns:
        wembed (String): word embedded text
    '''
    wembed = tf.keras.layers.Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=20, trainable=False)(x)
    return wembed

tweet = pd.read_csv('./data/tweets.csv', index_col=0)
review = pd.read_csv('./data/reviews.csv', usecols=['news_id', 'rating'])
tweet = pd.merge(tweet, review, on='news_id')
tweet.to_csv('./data/tweets_processed.csv')
max_length = 20

tweets = pd.read_csv('./data/tweets_processed.csv', usecols=['tweet.text', 'news_id', 'rating'])
tweets['verdict'] = tweets['rating'].apply(lambda x: int(x >= 3))
tweets['tweet.text'] = tweets['tweet.text'].progress_apply(lambda x: basic_cleanup(str(x)))

all_tweets = {}
for name, group in tweets.groupby('news_id'):
    all_tweets[name] = group['tweet.text'].values

tok = Tokenizer()
tok.fit_on_texts(tweets['tweet.text'])
vocab_size = len(tok.word_index) + 1
encoded_docs = tok.texts_to_sequences(tweets['tweet.text'])
tweet_padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
tweets['padded_tokens'] = list(tweet_padded_docs)

embeddings_index = dict()
f = open('./embeds/pretrained_embeddings.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((vocab_size, 100))
for word, i in tok.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

tweets['embeddings'] = tweets['padded_tokens'].progress_apply(lambda x: apply_embedding(x))

items = []
for name, group in tweets.groupby('news_id'):
    items.append({
        "news_id":name,
        "embeds": np.stack(group['embeddings'].values),
        "verdict": group['verdict'].values[0]
    })
    save(f'./data/tweets_embedded/{name}.npy', np.mean(np.stack(group['embeddings'].values), axis=0))

df = pd.DataFrame(items)
df['embeds'] = df['embeds'].apply(lambda x: np.mean(x, axis = 0))
df.to_csv('./data/agg_tweets_embeddings.csv')