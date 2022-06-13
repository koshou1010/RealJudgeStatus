import os 
import utils
import pandas as pd

INPUT_FOLDER = 'data'



if __name__ == '__main__':
    final_excel_name = 'statistic.xlsx'
    utils.copy_all_folder_path(INPUT_FOLDER)
    statistic_df = pd.DataFrame(columns= ['file', 'gas_in', 'reaction_stable', 'recovery'])
    for root, dirs, files in os.walk(INPUT_FOLDER):
        for file in files:
            if file.endswith('.ndjson'):
                rjs = utils.RealJudgeStatus(utils.load_parameters())
                statistic_df = rjs.dataload(os.path.join(root, file), statistic_df)
    # statistic_df.to_excel(final_excel_name, index = False)
    # statistic_df = pd.read_excel(final_excel_name)
    final_df = utils.calculate_statistic_result(statistic_df, utils.load_parameters())
    final_df.to_excel(final_excel_name, index = False)
    utils.reset_col_length(final_excel_name)

