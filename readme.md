# Real Judge Status

## How to Use

1. Setting data path in main.py, const called INPUT_FOLDER.
2. Put the ndjson file in INPUT_FOLDER. 
3. Open terminal and cd to current path, enter : python main.py

## Output Explain

1. RJS judge start from baseline stable flag of ESRTSD.
2. In the FIG_FOLDER, fig have two subfig in each .png file, one is ESRTSD control, anthoer is this tool judgment.
3. In the CSV_FOLDER, provide check (St2 - St1))/St1 of each file.
4. In the ERROR_FOLDER, was error file, would not add in statustic to calculate result.
5. statistic.xlsx is all of rjs sucess judge file, and can check each flag's time, so if there are something exception at dataload function will not store in statistic.xlsx.
6. In statistic.xlsx, each two rows are same file, the first one is ESRTSD control, the second one is tools judge.








"# RealJudgeStatus" 
