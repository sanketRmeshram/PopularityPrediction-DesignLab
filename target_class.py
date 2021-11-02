import pickle
from member_level_features import GetValidGrps
import pandas as pd
from pathlib import Path
def main():
    tag_new_slide = pickle.load(open("data/tag_new_slide1", "rb"))
    valid_windows = [str(x).split('/')[-1].split('.')[0].split('_') for x in Path("member_level_features_windowwise").iterdir()]
    for grp,window in valid_windows:
        grp_id = int(grp)
        window_index = int(window) 
        mp = {}
        mp["failure"] = [0]
        mp["neutral"] = [0]
        mp["success"] = [0]
        if tag_new_slide[grp_id][window_index] == -1:
            mp["failure"][0] =1
        elif tag_new_slide[grp_id][window_index] == 0:
            mp["neutral"][0] =1
        elif tag_new_slide[grp_id][window_index] == 1:
            mp["success"][0] =1
        df = pd.DataFrame(mp)
        df.to_csv("target_class_windowwise/"+str(grp_id) + "_" + str(window_index) + ".csv", index=False)
            
            
        

if __name__=="__main__":
    main()