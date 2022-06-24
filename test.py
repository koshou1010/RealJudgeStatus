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
    final_excel_name = 'statistic_test.xlsx'
    utils.copy_all_folder_path('data')
    file = r'data\food industrial\apple juice\0516\task__FT4JC60J__2022-05-16T12-33-41.102.ndjson'
    statistic_df = pd.DataFrame(columns= ['file', 'gas_in', 'reaction_stable', 'recovery'])
    features_df = pd.DataFrame()
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    log_filename = current_time+'__log.txt'
    rjs = utils.RealJudgeStatus(utils.load_parameters(), utils.Logger(log_filename))
    statistic_df, features_df = rjs.dataload(file, statistic_df, features_df)
    statistic_df.reset_index(drop = True, inplace = True)
    features_df.to_csv('features_test.csv')
    final_df = utils.calculate_statistic_result(statistic_df, utils.load_parameters())
    final_df.to_excel(final_excel_name, index = False)
    utils.reset_col_length(final_excel_name)
    


    
