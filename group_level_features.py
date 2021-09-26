import pickle

import pandas as pd
from math import log
import numpy as np
from pathlib import Path

from member_level_features import GetValidGrps,get_grp_to_valid_events,entropy,make_social_graph_of_grp_members
from geopy.distance import geodesic

# def average_distace_of_events(list_of_events,elat,elon) :
#     valid_list_of_events = [x for x in list_of_events if (x in elat and x in elon)]
#     n = len(valid_list_of_events)
#     total_dist = 0
#     cnt = 0
#     for i in range(n):
#         for j in range(i+1,n) :
#             x = (elat[valid_list_of_events[i]],elon[valid_list_of_events[i]])
#             y = (elat[valid_list_of_events[j]],elon[valid_list_of_events[j]])
#             total_dist += geodesic(x,y).miles
#             cnt+=1
#     if cnt==0 :
#         return 0
#     return total_dist/cnt
def average_variance_of_distace_of_events(list_of_events,elat,elon) :
    valid_list_of_events = [x for x in list_of_events if (x in elat and x in elon)]
    n = len(valid_list_of_events)
    total_dist = 0
    total_dist_sq = 0
    cnt = 0
    for i in range(n):
        for j in range(i+1,n) :
            x = (elat[valid_list_of_events[i]],elon[valid_list_of_events[i]])
            y = (elat[valid_list_of_events[j]],elon[valid_list_of_events[j]])
            dist = geodesic(x,y).miles
            total_dist += dist
            total_dist_sq += dist*dist
            cnt+=1
    if cnt==0 :
        return 0,0
    return total_dist/cnt,total_dist_sq/cnt - (total_dist/cnt)*(total_dist/cnt)

# def average_dist_member_event(list_of_events,elat,elon,ersvp,mlat,mlon):
#     cnt = 0
#     total_dist = 0
#     for event in list_of_events :
#         if event not in ersvp :
#             continue
#         for member,_ in ersvp[event] :
#             x = (elat[event],elon[event])
#             y = (mlat[member],mlon[member])
#             dist = geodesic(x,y).miles
#             total_dist += dist
#             cnt+=1
#     if cnt==0 :
#         return 0
#     return total_dist/cnt 
    
def average_variance_dist_member_event(list_of_events,elat,elon,ersvp,mlat,mlon):
    valid_list_of_events = [x for x in list_of_events if (x in elat and x in elon )]
    cnt = 0
    total_dist = 0
    total_dist_sq = 0
    for event in valid_list_of_events :
        if event not in ersvp :
            continue
        for member,_ in ersvp[event] :
            if not(member in mlat and member in mlon ):
                continue
            x = (elat[event],elon[event])
            y = (mlat[member],mlon[member])
            dist = geodesic(x,y).miles
            total_dist += dist
            total_dist_sq += dist*dist
            cnt+=1
    if cnt==0 :
        return 0,0
    return total_dist/cnt,total_dist_sq/cnt - (total_dist/cnt)*(total_dist/cnt)

# def average_of_distace_of_members(list_of_members,mlat,mlon) :
#     valid_list_of_members = [x for x in list_of_members if (x in mlat and x in mlon)]
#     n = len(valid_list_of_members)
#     total_dist = 0

#     cnt = 0
#     for i in range(n):
#         print(i)
#         for j in range(i+1,n) :
#             x = (mlat[valid_list_of_members[i]],mlon[valid_list_of_members[i]])
#             y = (mlat[valid_list_of_members[j]],mlon[valid_list_of_members[j]])
#             dist = geodesic(x,y).miles
# #             print("dissstance")
#             total_dist += dist
            
#             cnt+=1
#     if cnt==0 :
#         return 0
#     return total_dist/cnt

def average_variance_of_distace_of_members(list_of_members,mlat,mlon) :
    valid_list_of_members = [x for x in list_of_members if (x in mlat and x in mlon)]
    n = len(valid_list_of_members)
    total_dist = 0
    total_dist_sq = 0
    cnt = 0
    for i in range(n):
        for j in range(i+1,n) :
            x = (mlat[valid_list_of_members[i]],mlon[valid_list_of_members[i]])
            y = (mlat[valid_list_of_members[j]],mlon[valid_list_of_members[j]])
            dist = geodesic(x,y).miles
            total_dist += dist
            total_dist_sq += dist*dist
            cnt+=1
    if cnt==0 :
        return 0,0
    return total_dist/cnt,total_dist_sq/cnt - (total_dist/cnt)*(total_dist/cnt)

def get_avg_var_sum_rsvp(list_of_events,ersvp):
    total_rsvp = 0
    total_rsvp_sq = 0
    cnt = 0
    for event in list_of_events :
        cnt+=1
        if event not in ersvp :
            continue;
        temp = len(ersvp[event])
        total_rsvp += temp
        total_rsvp_sq += temp*temp
    if cnt == 0 :
        return 0,0,0
    return total_rsvp/cnt,total_rsvp_sq/cnt - (total_rsvp/cnt)*(total_rsvp/cnt),total_rsvp
            
    

def get_groups_features(group_level_features,grp_id,ersvp,groupjoin,new_grp_event,list_of_events,list_of_members,mlat,mlon,elat,elon) :

    group_level_features["group_id"] = []
    group_level_features["g1"] = []
    group_level_features["g2"] = []
    group_level_features["g3"] = []
    group_level_features["g4"] = []
    group_level_features["g5"] = []
    group_level_features["g6"] = []
    group_level_features["g7"] = []
    group_level_features["g8"] = []
    group_level_features["g9"] = []
    group_level_features["g10"] = []
    group_level_features["g11"] = []
    group_level_features["g12"] = []    
    group_level_features["g13"] = []
    group_level_features["g14"] = []  
    
    group_level_features["group_id"].append(grp_id)
    group_level_features["g1"].append(None)
    
    avg,var = average_variance_of_distace_of_events(list_of_events,elat,elon)
    group_level_features["g2"].append(avg)
    group_level_features["g3"].append(var)
    
    avg,var = average_variance_dist_member_event(list_of_events,elat,elon,ersvp,mlat,mlon)
    group_level_features["g4"].append(avg)
    group_level_features["g5"].append(var)

    avg,var = average_variance_of_distace_of_members(list_of_members,mlat,mlon)
    group_level_features["g6"].append(avg)
    group_level_features["g7"].append(var)
    
    entropy_of_location = entropy([(elat[x],elon[x]) for x in list_of_events if (x in elat and x in elon)])
    group_level_features["g8"].append(entropy_of_location)
    
    members,edges =  make_social_graph_of_grp_members(grp_id,groupjoin,ersvp,new_grp_event)
    density = 0
    if len(members) > 1 :
        density = (2*len(edges))/(len(members)*(len(members)-1))
    total_degree = 2*len(edges)
    group_level_features["g9"].append(density)
    group_level_features["g10"].append(total_degree)
    group_level_features["g11"].append(len(list_of_events))
    
    avg,var,tot = get_avg_var_sum_rsvp(list_of_events,ersvp)
    group_level_features["g12"].append(avg)
    group_level_features["g13"].append(var)
    group_level_features["g14"].append(tot)
    
    
    
    
    
def make_group_level_features(group_names,grp_to_valid_events,ersvp,groupjoin,new_grp_event,mlat,mlon,elat,elon) :
    group_level_features = {}

    group_level_features["group_id"] = []
    group_level_features["g1"] = []
    group_level_features["g2"] = []
    group_level_features["g3"] = []
    group_level_features["g4"] = []
    group_level_features["g5"] = []
    group_level_features["g6"] = []
    group_level_features["g7"] = []
    group_level_features["g8"] = []
    group_level_features["g9"] = []
    group_level_features["g10"] = []
    group_level_features["g11"] = []
    group_level_features["g12"] = []    
    group_level_features["g13"] = []
    group_level_features["g14"] = []    
    
    y = set([int(str(x).split('/')[-1].split('.')[0]) for x in Path("group_level_features").iterdir()])
    tot = 0
    ind = 1
    for grp_id in group_names:
        if grp_id in y:
            print(ind)
            ind+=1
            continue
        df = pd.read_csv("member_level_features/"+str(grp_id) + ".csv")
        if grp_id in grp_to_valid_events :     
            get_groups_features(group_level_features,grp_id,ersvp,groupjoin,new_grp_event,grp_to_valid_events[grp_id],list(df['member']),mlat,mlon,elat,elon)
        else :
            get_groups_features(group_level_features,grp_id,ersvp,groupjoin,new_grp_event,[],list(df['member']),mlat,mlon,elat,elon)
            tot+=1
        print(ind)
        ind+=1
        df = pd.DataFrame(group_level_features)
        df.to_csv("group_level_features/"+ str(grp_id) +".csv",index=False)
    print("****** \n \n\n   " ,tot,"  \n\n\n\n\n  ********** ")
        
#     df = pd.DataFrame(group_level_features)
#     df.to_csv("group_level_features/group_level_features.csv",index=False)
    
    
        
        
    
    

def main():
    
    
    group_names = [int(str(x).split('/')[-1].split('.')[0]) for x in Path("member_level_features").iterdir()]
    elat = pickle.load(open("data/elat","rb"))
    elon = pickle.load(open("data/elon","rb"))
    mlat = pickle.load(open("data/mlat","rb"))
    mlon = pickle.load(open("data/mlon","rb"))
    groupjoin = pickle.load(open("data/groupjoin","rb"))
    new_grp_event = pickle.load(open("data/new_grp_event","rb"))
    ersvp = pickle.load(open("data/ersvp","rb"))
    valid_grps = GetValidGrps(new_grp_event,groupjoin)
    grp_to_valid_events=get_grp_to_valid_events(valid_grps,new_grp_event,ersvp)

    make_group_level_features(group_names,grp_to_valid_events,ersvp,groupjoin,new_grp_event,mlat,mlon,elat,elon)
    
    

if __name__=="__main__":
    main()