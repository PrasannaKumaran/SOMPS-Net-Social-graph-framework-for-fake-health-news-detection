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

max_length = 20
retweet = pd.read_csv('./data/retweets.csv', index_col=0)
review = pd.read_csv('./data/reviews.csv', usecols=['news_id', 'rating'])
retweet = pd.merge(retweet, review, on='news_id')
retweet.to_csv('./data/retweets_processed.csv')

retweets = pd.read_csv('./data/retweets_processed.csv', usecols=['retweet.text', 'news_id', 'rating'])
retweets['verdict'] = retweets['rating'].apply(lambda x: int(x >= 3))
retweets['retweet.text'] = retweets['retweet.text'].apply(lambda x: basic_cleanup(str(x)))

all_retweets = {}
for name, group in retweets.groupby('news_id'):
    all_retweets[name] = group['retweet.text'].values

tok = Tokenizer()
tok.fit_on_texts(retweets['retweet.text'])
vocab_size = len(tok.word_index) + 1
encoded_docs = tok.texts_to_sequences(retweets['retweet.text'])
retweet_padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
retweets['padded_tokens'] = list(retweet_padded_docs)

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

retweets['embeddings'] = retweets['padded_tokens'].progress_apply(lambda x: apply_embedding(x))
retweets.to_csv("./data/retweets_embeds.csv")

items = []
for name, group in retweets.groupby('news_id'):
    items.append({
        "news_id":name,
        "embeds": np.stack(group['embeddings'].values),
        "verdict": group['verdict'].values[0]
    })
    save(f'./retweet_texts_embedded/{name}.npy', np.mean(np.stack(group['embeddings'].values), axis=0))

df = pd.DataFrame(items)
df['embeds'] = df['embeds'].apply(lambda x: np.mean(x, axis = 0))
df.to_csv('./data/agg_retweets_embeddings.csv')