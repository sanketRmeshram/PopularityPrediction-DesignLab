import pickle

import pandas as pd
from math import log
import numpy as np
from pathlib import Path

from geopy.distance import geodesic
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

def average_variance_dist_member_event(list_of_events,elat,elon,list_of_members,mlat,mlon):
    valid_list_of_events = [x for x in list_of_events if (x in elat and x in elon )]
    cnt = 0
    total_dist = 0
    total_dist_sq = 0
    for event in valid_list_of_events :
        for member in list_of_members :
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
            
def entropy(a):
    total = len(a)
    freq = {}
    for i in a:
        if i in freq:
            freq[i] += 1
        else:
            freq[i] = 1
    ans = 0
    for i in freq:
        ans += -(freq[i]/total) * log((freq[i]/total), 2)
    return ans


    
def get_graph(grp_id,window) :
    graph_file = open("graph_windowwise/"+str(grp_id) +"_" + str(window) + ".txt","r")
    line = graph_file.readline()
    list_of_members = [ int(x) for x in line.split() ]
    adj = {}
    edges = []
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
        edges.append((i,j,w))
        adj[i].append((j,w))
        adj[j].append((i,w))
    return list_of_members,adj,edges
            

def main():
    
    valid_windows = [str(x).split('/')[-1].split('.')[0].split('_') for x in Path("member_level_features_windowwise").iterdir()]
    elat = pickle.load(open("data/elat","rb"))
    elon = pickle.load(open("data/elon","rb"))
    mlat = pickle.load(open("data/mlat","rb"))
    mlon = pickle.load(open("data/mlon","rb"))
    groupjoin = pickle.load(open("data/groupjoin","rb"))
    new_grp_event = pickle.load(open("data/new_grp_event","rb"))
    ersvp = pickle.load(open("data/ersvp","rb"))
    
    for grp,window in valid_windows:
        grp_id = int(grp)
        window_index = int(window) 
        list_of_events = new_grp_event[grp_id][window_index:window_index+10]
        list_of_members,adj,edges = get_graph(grp_id,window)
        
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
        
        group_level_features["group_id"].append(grp_id)
        group_level_features["g1"].append(None)
        
        avg,var = average_variance_of_distace_of_events(list_of_events,elat,elon)
        group_level_features["g2"].append(avg)
        group_level_features["g3"].append(var)
        
        avg,var = average_variance_dist_member_event(list_of_events,elat,elon,list_of_members,mlat,mlon)
        group_level_features["g4"].append(avg)
        group_level_features["g5"].append(var)
        
        avg,var = average_variance_of_distace_of_members(list_of_members,mlat,mlon)
        group_level_features["g6"].append(avg)
        group_level_features["g7"].append(var)
    
        entropy_of_location = entropy([(elat[x],elon[x]) for x in list_of_events if (x in elat and x in elon)])
        group_level_features["g8"].append(entropy_of_location)
        density = 0
        if len(list_of_members) > 1 :
            density = (2*len(edges))/(len(list_of_members)*(len(list_of_members)-1))
        total_degree = 2*len(edges)
        group_level_features["g9"].append(density)
        group_level_features["g10"].append(total_degree)
        group_level_features["g11"].append(len(list_of_events))
        
        avg,var,tot = get_avg_var_sum_rsvp(list_of_events,ersvp)
        group_level_features["g12"].append(avg)
        group_level_features["g13"].append(var)
        group_level_features["g14"].append(tot)
        
        df = pd.DataFrame(group_level_features)
        df.to_csv("group_level_features_windowwise/"+ str(grp_id)+"_"+str(window) +".csv",index=False)
        
        print(grp_id,window)

if __name__=="__main__":
    main()