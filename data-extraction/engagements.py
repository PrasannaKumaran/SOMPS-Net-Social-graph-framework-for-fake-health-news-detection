import os
import json
import pandas as pd
import pickle as pkl
from pandas.io.json import json_normalize

tweets_columns = pkl.load(open("tweet_columns.pkl", "rb"))
retweets_columns = pkl.load(open("retweet_columns.pkl", "rb"))
reply_columns = pkl.load(open("reply_columns.pkl", "rb"))   

def extract_engagements(dataset):
    '''
    Extracts and stores the engagements (tweet, retweet, replies) from the dataset
    Parameters:
        dataset (String): Name of the dataset
    Returns:
        None
    '''
    reviews_path = f"./engagements/{dataset}/"
    tweet_data_rows = []
    retweet_data_rows = []
    replies_data_rows = []
    
    print(f"******************************* {dataset}********************************")
    for i, review in enumerate(os.listdir(reviews_path)):
        print(i)
        if not review.startswith('.'):
            tweet_path = os.path.join(reviews_path, review, "tweets")
            retweets_path = os.path.join(reviews_path, review, "retweets")
            replies_path = os.path.join(reviews_path, review, "replies")

            for tweet in os.listdir(tweet_path):
                if '.json' in tweet:
                    tweet_data = json_normalize(json.load(open(os.path.join(tweet_path, tweet), "r")))
                    tweet_extract = {}
                    for column in tweets_columns:
                        if column in tweet_data.columns:
                            tweet_extract['tweet.' + column] = tweet_data[column].values[0]
                    tweet_extract['tweet.user_id'] = tweet_data['user.id'].values[0]
                    tweet_extract['news_id'] = review
                    tweet_data_rows.append(tweet_extract)            

            for retweet in os.listdir(retweets_path):
                if '.json' in retweet:
                    retweet_data = json_normalize(json.load(open(os.path.join(retweets_path, retweet), "r")))
                    retweet_extract = {}
                    for column in retweets_columns:
                        if column in retweet_data.columns:
                            retweet_extract['retweet.' + column] = retweet_data[column].values[0]
                    retweet_extract['retweet.user_id'] = retweet_data['user.id'].values[0]
                    retweet_extract['news_id'] = review
                    retweet_data_rows.append(retweet_extract)

            for reply in os.listdir(replies_path):
                reply_data = json_normalize(json.load(open(os.path.join(replies_path, reply), "r")))
                reply_extract = {}
                for column in reply_columns:
                     if column in reply_data.columns:
                         reply_extract['reply.' + column] = reply_data[column].values[0]
                reply_extract['reply.user_id'] = reply_data['user.id'].values[0]
                reply_extract['news_id'] = review
                replies_data_rows.append(reply_extract)
      
    tweets_ = pd.DataFrame(tweet_data_rows)        
    retweets_ = pd.DataFrame(retweet_data_rows)
    replies_ = pd.DataFrame(replies_data_rows)
    tweets_.to_csv(f"./data/{dataset}_tweets.csv")
    retweets_.to_csv(f"./data/{dataset}_retweets.csv")
    replies_.to_csv(f"./data/{dataset}_replies.csv")
   
dataset = ['HealthStory', 'HealthRelease']
for d in dataset:
    extract_engagements(d)
