import os
import utils
import json
import pandas as pd



def load_parameters() -> (dict):
    with open('parameters.json') as json_file:
        data = json.load(json_file) 
    return data

if __name__ == '__main__':
    file = 'data\\task__SXT273W5NHA__2022-04-20T17-18-01.788.ndjson'
    statistic_df = pd.DataFrame(columns= ['file', 'gas_in', 'reaction_stable', 'recovery'])
    rjs = utils.RealJudgeStatus(utils.load_parameters())
    statistic_df = rjs.dataload(file, statistic_df)


    
