import os
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from numpy import save
tqdm.pandas()

tweets = pd.read_csv("./data/tweets.csv", index_col=0)
review = pd.read_csv('./data/reviews.csv', index_col=0)
reviews = list(set(review['news_id']))

user_followers_path = './data/user_followers_list/'
followers_profile_ids_available = os.listdir(user_followers_path)
followers_profile_ids_available = [file.split('.')[0] for file in followers_profile_ids_available]

user_following_path = './data/user_following_list/'
following_profile_ids_available = os.listdir(user_following_path)
following_profile_ids_available = [file.split('.')[0] for file in following_profile_ids_available]

common_ids = list(set(followers_profile_ids_available).intersection(following_profile_ids_available))
tweets['tweet.user_id'] = tweets['tweet.user_id'].apply(lambda x: str(int(float(x))))

def tweet_users_similarity():
    '''
    Calculates similarity between users who tweeted 
    on the article based on followers and followings
    
    Parameters:
        None
    Returns:
        None
    '''
    missed_news = []
    for news in tqdm(reviews):
        users = tweets[tweets['news_id'] == news][['tweet.user_id', 'tweet.created_at']].sort_values(by=['tweet.created_at'], ascending=False)
        users = users[users['tweet.user_id'].isin(common_ids)]
        if users.shape[0] > 0:
            if users.shape[0] < 118:
                users = users.sample(n=118, replace=True)
            select_users = []
            
            for user in users['tweet.user_id']:
                select_users.append(user)
                if len(select_users) == 118:
                    break
            with open (f'./data/tweets_network_similarity_user_list/{news}.pkl', 'wb') as f:
                pkl.dump(select_users, f)
            similarity_matrix = []
            
            try:
                for i in range(len(select_users)):
                    user_similarity = []
                    for j in range(len(select_users)):
                        user1_followers = pkl.load(open(f'./data/user_followers_list/{str(select_users[i])}.pkl', 'rb'))
                        user2_followers = pkl.load(open(f'./data/user_followers_list/{str(select_users[j])}.pkl', 'rb'))
                        common_followers = set(user1_followers).intersection(user2_followers)

                        user1_followings = pkl.load(open(f'./data/user_following_list/{str(select_users[i])}.pkl', 'rb'))
                        user2_followings = pkl.load(open(f'./data/user_following_list/{str(select_users[j])}.pkl', 'rb'))
                        common_followings = set(user1_followings).intersection(user2_followings)

                        ration_common_network = (len(common_followers.union(common_followings))) / (len(set(user1_followers + user2_followers + user1_followings + user2_followings)) + 1e-6) 
                        user_similarity.append(ration_common_network)    

                    similarity_matrix.append(user_similarity)
                save(f'./data/tweets_network_similarity/{news}_user_similarity_matrix.npy', similarity_matrix)
            except FileNotFoundError:
                missed_news.append(news)

    with open('./tweets_missed_news.pkl', 'wb') as f:
        pkl.dump(missed_news, f)
    
tweet_users_similarity()    
