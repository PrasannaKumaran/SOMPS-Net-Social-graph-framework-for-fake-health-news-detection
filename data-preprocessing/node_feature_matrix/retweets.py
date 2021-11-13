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
reqd_columns = pkl.load(open("retweet_required_columns.pkl", "rb"))

similarity_matrix_retweet = []
for news in tqdm(required_articles):
    value = load(f'./data/retweets_network_similarity/{news}_user_similarity_matrix.npy')
    similarity_matrix_retweet.append(value)

similarity_matrix_retweet = np.array(similarity_matrix_retweet)
graph_conv_filters_retweet = preprocess_adj_tensor(similarity_matrix_retweet)
save('./retweet_data/graph_retweet_data_network_similarity.npy', graph_conv_filters_retweet)

retweets = pd.read_csv("./data/retweets.csv", lineterminator='\n', index_col=0)
story = pd.read_csv("./data/story.csv")

news_articles = sorted(story['story_review_no'])
retweets['retweet.user_id'] = retweets['retweet.user_id'].apply(lambda x : str(int(float(x))))
retweets.columns

news_retweet_info = {}
for news in tqdm(required_articles):
    news_retweet_info[news] = []
    required_users = pkl.load(open(f'./data/retweets_network_similarity_user_list/{news}.pkl', 'rb'))
    current_news_df = retweets[(retweets['news_id'] == news) & (retweets['retweet.user_id'].isin(required_users))]
    for user in required_users:
        current_user = current_news_df.loc[retweets['retweet.user_id'] == user]
        data = []
        for col in reqd_columns:
            data.append(float(current_user[col].values[0]))
        news_retweet_info[news].append(data)
        
for news in news_retweet_info.keys():
    with open(f'./retweet_data/retweet_user_features/{news}.pkl', 'wb') as f:
        pkl.dump(news_retweet_info[news], f)

graph_retweet_data = []
for news in required_articles:
    data = pkl.load(open(f'./retweet_data/retweet_user_features/{news}.pkl', 'rb'))
    data = np.asarray(data)
    data = data.tolist()
    graph_retweet_data.append(data)
save('./retweet_data/graph_retweet_data.npy', graph_retweet_data)
