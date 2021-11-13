import os
import json
import pandas as pd
import pickle as pkl

health_story_path = "./HealthStory"
health_story_dir = os.listdir(path=health_story_path)

main_cols = pkl.load(open("./main_cols.pkl", 'rb')) 
column_list = pkl.load(open("./columns.pkl", 'rb'))

def contents_extraction(health_story_dir):
    '''
    Extracts and stores contents of the news article.
    Parameter:
        health_story_dir (String) : directory path
    Returns:
        None
    '''
    rows = []   
    for file in health_story_dir:
        try: 
            data = json.load(open(health_story_path + '/' + file))
            row_data = list(map(data.get, main_cols))
            if 'meta_data' in data.keys():
                if 'og' in data['meta_data'].keys():
                    add_on = [data['meta_data']['og'][key] if key in data['meta_data']['og'].keys() else None for key in ['description', 'site_name']]
                    row_data.extend(add_on)
                    row_data.append(file.split('.')[0])
                else:
                    row_data.extend([data['meta_data']['description'], None, file.split('.')[0]])  
        except KeyError:
            data = json.load(open(health_story_path + '/' + file))
            row_data = list(map(data.get, main_cols))
            row_data.extend([None, None, file.split('.')[0]])
        except IsADirectoryError:
            continue
        finally: 
            rows.append(row_data)

    health_story_df = pd.DataFrame(rows, columns=column_list)
    health_story_df.to_csv("./data/HealthStory.csv")

contents_extraction(health_story_dir)