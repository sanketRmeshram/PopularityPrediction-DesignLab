import pandas as pd
import torch
from pathlib import Path

def get_graph(grp_id:int,window:int) :
    graph_file = open("graph_windowwise/"+str(grp_id) +"_" + str(window) + ".txt","r")
    line = graph_file.readline()
    list_of_members = [ int(x) for x in line.split() ]
    adj = {}
    for member in list_of_members:
        adj[member] = []
    while True:
        line = graph_file.readline()
        if not line :
            break
        i,j,w = line.split()
        i = int(i)
        j = int(j)
        w = int(w)
        adj[i].append(j)
        adj[j].append(i)
    return adj

def get_output(grp_id:int,window:int):
    df = pd.read_csv("target_class_windowwise/"+str(grp_id) + "_" + str(window) + ".csv")
    return torch.tensor(df.iloc[0,:])

def get_member_role_vectors(grp_id:int,window:int):
    df = pd.read_csv("NMF_member_level_features_windowwise/"+str(grp_id) + "_" + str(window) + ".csv")
    ans = {}
    features_cols = list(df.columns)[2:]
    for i in range(len(df)) :
        ans[int(df.iloc[i,0])] =  torch.tensor( df.iloc[i,:][features_cols]).float()
        
    return ans
        
def get_group_level_featues(grp_id:int,window:int):
    df = pd.read_csv("group_level_features_windowwise/"+str(grp_id) + "_" + str(window) + ".csv")
    df.iloc[0,1] = 0
    feature_cols = list(df.columns)[1:]
    return torch.tensor(df.iloc[0,:][feature_cols] ).float()
    
    
def get_group_and_window() :
    valid_windows = [str(x).split('/')[-1].split('.')[0].split('_') for x in Path("member_level_features_windowwise").iterdir()]
    return [(int(grp_id),int(window)) for grp_id,window in valid_windows]


if __name__=="__main__":
    now = get_group_and_window()
    print(len(now))
    grp,window = now[0]
    print("graph : " , get_graph(grp,window))
    print( "output : ",get_output(grp,window))
    print("member role  : ",get_member_role_vectors(grp,window))
    print("group level features : ",get_group_level_featues(grp,window))

    
    
    