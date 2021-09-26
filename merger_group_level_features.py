import pandas as pd

from pathlib import Path

def main():
    group_names = [int(str(x).split('/')[-1].split('.')[0]) for x in Path("group_level_features").iterdir()]
    dfs = []
    for grp_id in group_names :
        df = pd.read_csv("group_level_features/"+str(grp_id) + ".csv")
        dfs.append(df)
    result = pd.concat(dfs)
    result.to_csv("group_level_features.csv",index = False)
        
    

if __name__=="__main__":
    main()