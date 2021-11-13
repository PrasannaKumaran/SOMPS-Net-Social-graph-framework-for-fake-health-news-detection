import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
from numpy import save, load
tqdm.pandas()

def normalize_adj_numpy(adj, symmetric=True):
    if symmetric:
        d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d)
    else:
        d = np.diag(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj)
    return a_norm

def preprocess_adj_tensor(adj_tensor, symmetric=True):
    adj_out_tensor = []
    for i in range(adj_tensor.shape[0]):
        adj = adj_tensor[i]
        adj = adj + np.eye(adj.shape[0])
        adj = normalize_adj_numpy(adj, symmetric)        
        adj_out_tensor.append(adj)
    adj_out_tensor = np.array(adj_out_tensor)
    return adj_out_tensor

required_articles =pkl.load(open('./data/required_articles.pkl', 'rb'))
reqd_columns = pkl.load(open("tweet_required_columns.pkl", "rb"))

similarity_matrix_tweet = []
for news in tqdm(required_articles):
    value = load(f'./data/tweets_network_similarity/{news}_user_similarity_matrix.npy')
    similarity_matrix_tweet.append(value)

similarity_matrix_tweet = np.array(similarity_matrix_tweet)
graph_conv_filters_tweet = preprocess_adj_tensor(similarity_matrix_tweet)
save('./tweet_data/graph_tweet_data_network_similarity.npy', graph_conv_filters_tweet)

tweets = pd.read_csv("./data/tweets.csv", lineterminator='\n', index_col=0)
story = pd.read_csv("./data/story.csv")
news_articles = sorted(story['story_review_no'])

tweets['tweet.user_id'] = tweets['tweet.user_id'].apply(lambda x : str(int(float(x))))

news_tweet_info = {}
for news in tqdm(required_articles):
    news_tweet_info[news] = []
    required_users = pkl.load(open(f'../FakeHealth /data/HS_tweets_network_similarity_user_list/{news}.pkl', 'rb'))
    current_news_df = tweets[(tweets['news_id'] == news) & (tweets['tweet.user_id'].isin(required_users))]
    for user in required_users:
        current_user = current_news_df.loc[tweets['tweet.user_id'] == user]
        data = []
        for col in reqd_columns:
            data.append(float(current_user[col].values[0]))
        news_tweet_info[news].append(data)

for news in news_tweet_info.keys():
    with open(f'./tweet_data/tweet_user_features/{news}.pkl', 'wb') as f:
        pkl.dump(news_tweet_info[news], f)

graph_tweet_data = []
for news in required_articles:
    data = pkl.load(open(f'./tweet_data/tweet_user_features/{news}.pkl', 'rb'))
    data = np.asarray(data)
    data = data.tolist()
    graph_tweet_data.append(data)
save('./tweet_data/graph_tweet_data.npy', graph_tweet_data)