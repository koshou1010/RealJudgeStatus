import os
import utils
import json
import pandas as pd
import datetime


def load_parameters() -> (dict):
    with open('parameters.json') as json_file:
        data = json.load(json_file) 
    return data

if __name__ == '__main__':
    file = r'data\food industrial\apple juice\0513\task__FT4JC60J__2022-05-13T09-37-46.867.ndjson'
    statistic_df = pd.DataFrame(columns= ['file', 'gas_in', 'reaction_stable', 'recovery'])
    features_df = pd.DataFrame()
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    log_filename = current_time+'__log.txt'
    rjs = utils.RealJudgeStatus(utils.load_parameters(), utils.Logger(log_filename))
    statistic_df, features_df = rjs.dataload(file, statistic_df, features_df)
    


    
