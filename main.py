import os 
import utils
import pandas as pd
import datetime

INPUT_FOLDER = 'data'



if __name__ == '__main__':
    final_excel_name = 'statistic.xlsx'
    utils.copy_all_folder_path(INPUT_FOLDER)
    statistic_df = pd.DataFrame(columns= ['file', 'gas_in', 'reaction_stable', 'recovery'])
    features_df = pd.DataFrame()
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    log_filename = current_time+'__log.txt'
    for root, dirs, files in os.walk(INPUT_FOLDER):
        for file in files:
            if file.endswith('.ndjson'):
                rjs = utils.RealJudgeStatus(utils.load_parameters(), utils.Logger(log_filename))
                statistic_df, features_df = rjs.dataload(os.path.join(root, file), statistic_df, features_df)
    statistic_df.reset_index(drop = True, inplace = True)
    features_df.to_csv('features.csv')
    final_df = utils.calculate_statistic_result(statistic_df, utils.load_parameters())
    final_df.to_excel(final_excel_name, index = False)
    utils.reset_col_length(final_excel_name)

