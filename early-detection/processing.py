import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from numpy import load, save
tqdm.pandas()

reviews = pd.read_csv('../data/reviews.csv')
articles = pkl.load(open('../data/article_list.pkl', 'rb'))
common_ids = pkl.load(open('../data/common_ids.pkl', 'rb'))
reviews['verdict'] = reviews['rating'].apply(lambda x: int(x >= 3))

reqd_columns_tweets = pkl.load(open("./tweet_columns.pkl", 'rb'))
reqd_columns_retweets = pkl.load(open("./retweet_columns.pkl", 'rb'))

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

def similarity_calculation(data, engagement, required_news_ids, threshold, data_path_folder):
    missed_news = []
    for news in tqdm(required_news_ids):
        users = data[data['news_id_y'] == str(news)][[
            f'{engagement}.user_id',
            f'{engagement}.created_at']].sort_values(by=[f'{engagement}.created_at'], ascending=False)
        users = users[users[f'{engagement}.user_id'].isin(common_ids)]
        if users.shape[0] > 0:
            if users.shape[0] < threshold:
                users = users.sample(n=threshold, replace=True)
            select_users = []

            for user in users[f'{engagement}.user_id']:
                if str(user) in common_ids:
                    select_users.append(user)
                if len(select_users) == threshold:
                    break
            
            with open (f'{data_path_folder}Network_{engagement}/network_similarity_user_list_{news}.pkl', 'wb') as f:
                pkl.dump(select_users, f)
            similarity_matrix = []
            for i in range(len(select_users)):
                user_similarity = []
                for j in range(len(select_users)):
                    user1_followers = pkl.load(open(f'../../FakeHealth /data/user_followers_list/{str(select_users[i])}.pkl', 'rb'))
                    user2_followers = pkl.load(open(f'../../FakeHealth /data/user_followers_list/{str(select_users[j])}.pkl', 'rb'))
                    common_followers = set(user1_followers).intersection(user2_followers)

                    user1_followings = pkl.load(open(f'../../FakeHealth /data/user_following_list/{str(select_users[i])}.pkl', 'rb'))
                    user2_followings = pkl.load(open(f'../../FakeHealth /data/user_following_list/{str(select_users[j])}.pkl', 'rb'))
                    common_followings = set(user1_followings).intersection(user2_followings)

                    ration_common_network = (len(common_followers.union(common_followings))) / (len(set(user1_followers + user2_followers + user1_followings + user2_followings)) + 1e-6) 
                    user_similarity.append(ration_common_network)  

                similarity_matrix.append(user_similarity)
            save(f'{data_path_folder}/Network_{engagement}/{news}_user_similarity_matrix.npy', similarity_matrix)
            
    with open(f'{data_path_folder}/Network_{engagement}/Missed_news_{engagement}.pkl', 'wb') as f:
        pkl.dump(missed_news, f)
        
def feature_matrix(articles, data_path_folder, engagement, dataframe, threshold, required_columns):

    similarity_matrix_tweet = []
    for news in tqdm(articles):
        value = load(f'{data_path_folder}Network_{engagement}/{news}_user_similarity_matrix.npy')
        similarity_matrix_tweet.append(value)

    similarity_matrix_tweet = np.array(similarity_matrix_tweet)
    graph_conv_filters_tweet = preprocess_adj_tensor(similarity_matrix_tweet)
    save(f'{data_path_folder}graph_{engagement}_data_network_similarity.npy', graph_conv_filters_tweet)

    news_data_info = {}
    for news in tqdm(articles):
        news_data_info[news] = []
        required_users = pkl.load(open(f'{data_path_folder}Network_{engagement}/network_similarity_user_list_{news}.pkl', 'rb'))
        current_user_df = dataframe[(dataframe['news_id_y'] == news) & (dataframe[f'{engagement}.user_id'].isin(required_users))]
        for user in required_users:
            current_user = current_user_df.loc[dataframe[f'{engagement}.user_id'] == user]
            data = []
            for col in required_columns:
                data.append(float(current_user_df[col].values[0]))
            news_data_info[news].append(data)

    for news in news_data_info.keys():
        with open(f'{data_path_folder}{engagement}_user_features/{news}.pkl', 'wb') as f:
            pkl.dump(news_data_info[news], f)

    graph_data = []
    tweet_user_size = threshold
    for news in articles:
        data = pkl.load(open(f'{data_path_folder}{engagement}_user_features/{news}.pkl', 'rb'))
        data = np.asarray(data)
        data = data.tolist()
        graph_data.append(data)
    save(f'{data_path_folder}graph_{engagement}_data.npy', graph_data)

def threshold_estimation(data):
    values = []    
    for name, group in data.groupby('news_id_y'):
        values.append(group.shape[0])
    return int(np.median(values))

def labels(articles, data_path):
    labels = []
    for news in articles:
        current_review = reviews[reviews['news_id'] == news]['verdict'].values[0]
        labels.append(current_review)
    save(f'{data_path}labels.npy', labels)

def processing(tweet_path_folder, retweet_path_folder, dataset_hours):
    tweets = pd.read_csv(f'{tweet_path_folder}/{dataset_hours}_tweets.csv', index_col=0)
    retweets = pd.read_csv(f'{retweet_path_folder}/{dataset_hours}_retweets.csv', index_col=0)
    tweets = tweets[tweets['news_id_y'].isin(articles)]
    retweets = retweets[retweets['news_id_y'].isin(articles)]
    
    tweets_news_items = set(tweets['news_id_y'].values)
    retweets_news_items = set(retweets['news_id_y'].values)
    required_news_ids = tweets_news_items.intersection(retweets_news_items)
    
    with open(f'./data/{dataset_hours}_required_news_articles.pkl', 'wb') as f:
        pkl.dump(required_news_ids, f)
    
    tweets['tweet.user_id'] = tweets['tweet.user_id'].apply(lambda x : str(int(float(x))))
    retweets['retweet.user_id'] = retweets['retweet.user_id'].apply(lambda x : str(int(float(x))))
    
    threshold_tweet = threshold_estimation(tweets)
    threshold_retweet = threshold_estimation(retweets)
    similarity_calculation(tweets, 'tweet', required_news_ids, threshold_tweet, tweet_path_folder)
    similarity_calculation(retweets, 'retweet', required_news_ids, threshold_retweet, retweet_path_folder)

    required_columns_tweet = reqd_columns_tweets
    required_columns_retweet = reqd_columns_retweets
    
    feature_matrix(required_news_ids, tweet_path_folder, 'tweet', tweets, threshold_tweet, required_columns_tweet)
    feature_matrix(required_news_ids, retweet_path_folder, 'retweet', retweets, threshold_retweet, required_columns_retweet)
    
    labels(required_news_ids, tweet_path_folder)

hours = [4, 8, 12, 16, 20, 24]
for i in tqdm(hours):
    dataset_hours = f'HS{i}h'
    tweet_path_folder = f'./data/HealthStory_{i}_Hours/'
    retweet_path_folder = f'./data/HealthStory_{i}_Hours/'
    processing(tweet_path_folder, retweet_path_folder, dataset_hours)