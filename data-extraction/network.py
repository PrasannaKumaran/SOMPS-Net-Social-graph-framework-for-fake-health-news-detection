import os
import json
import pandas as pd
import pickle as pkl
from pandas.io.json import json_normalize

profile_columns = pkl.load(open("profile_columns.pkl", "rb"))

def extract_network():
    '''
    Extract and store the network details (followers, following, profiles) of the dataset
    Parameters:
        None
    Returns:
        None
    '''
    followers_data_rows = []
    following_data_rows = []
    profile_data_rows = []

    user_followers_path = f'./user_network/user_followers/'
    user_following_path = f'./user_network/user_following/'
    user_profiles_path = f'./user_network/user_profiles/'

    try: 
        for followers in os.listdir(user_followers_path):
            followers_data = json_normalize(json.load(open(os.path.join(user_followers_path, followers), "r")))
            followers_extract = {
                "id": followers.split('.')[0],
                "number_of_followers": len(followers_data['ids'].values[0])
            } 
            followers_data_rows.append(followers_extract)
            
        for following in os.listdir(user_following_path):
            following_data = json_normalize(json.load(open(os.path.join(user_following_path, following), "r")))
            following_extract = {
                "id": following.split('.')[0],
                "number_of_followings": len(following_data['ids'].values[0])
            } 
            following_data_rows.append(following_extract)
            
        for profile in os.listdir(user_profiles_path):
            profile_data = json_normalize(json.load(open(os.path.join(user_profiles_path, profile), "r")))
            profile_extract = {}
            for column in profile_columns:
                if column in profile_data.columns:
                    profile_extract['user.' + column] = profile_data[column].values[0]
            profile_data_rows.append(profile_extract)
            
    except IsADirectoryError:
        pass

    followers_ = pd.DataFrame(followers_data_rows)        
    following_ = pd.DataFrame(following_data_rows)
    profile_ = pd.DataFrame(profile_data_rows)
    
    followers_.to_csv("./data/user_followers.csv")
    following_.to_csv("./data/user_following.csv")
    profile_.to_csv("./data/user_profile.csv")

extract_network()