import os
import utils
import json
import pandas as pd



def load_parameters() -> (dict):
    with open('parameters.json') as json_file:
        data = json.load(json_file) 
    return data

if __name__ == '__main__':
    file = 'data\\20220420\\DMMP\\task__SXT273W5NHA__2022-04-20T12-49-36.389.ndjson'
    statistic_df = pd.DataFrame(columns= ['file', 'gas_in', 'reaction_stable', 'recovery'])
    FIG_FOLDER = 'fig'
    rjs = utils.RealJudgeStatus(FIG_FOLDER, load_parameters())
    rjs.test_dataload(file)
    # rjs.statistic_total(statistic_df)

    
