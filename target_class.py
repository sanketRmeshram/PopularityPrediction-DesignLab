import pickle
from member_level_features import GetValidGrps
import pandas as pd

def main():
    groupjoin = pickle.load(open("data/groupjoin", "rb"))
    new_grp_event = pickle.load(open("data/new_grp_event", "rb"))
    ersvp = pickle.load(open("data/ersvp", "rb"))
    valid_grps = GetValidGrps(new_grp_event, groupjoin)
    tag_new_slide = pickle.load(open("data/tag_new_slide1", "rb"))
    valid_grps = set(valid_grps)
    valid_grps = set([i for i in valid_grps if i in tag_new_slide])
    print("valid_grps : " , len(valid_grps))
    for grp in valid_grps:
        print(grp)
        now = tag_new_slide[grp]
        for i in range(len(now)-1):
            mp = {}
            mp["failure"] = [0]
            mp["neutral"] = [0]
            mp["success"] = [0]
            if now[i] == -1:
                mp["failure"][0] =1
            elif now[i] == 0:
                mp["neutral"][0] =1
            elif now[i] == 1:
                mp["success"][0] =1
            df = pd.DataFrame(mp)
            df.to_csv("target_class_windowwise/"+str(grp) + "_" + str(i) + ".csv", index=False)
            
            
        

if __name__=="__main__":
    main()