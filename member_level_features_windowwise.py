
import pickle

import pandas as pd
from math import log
import numpy as np
from sklearn.decomposition import NMF


def make_social_graph_of_grp_members(grp_id, groupjoin, ersvp, new_grp_event):
    member_set = set()
    for member, time in groupjoin[grp_id]:
        member_set.add(member)

    events_in_grp = []

    for event_id, time in new_grp_event[grp_id]:
        if event_id in ersvp:
            events_in_grp.append(event_id)
    edges = {}
    for event_id in events_in_grp:

        event_member_set = list(
            set([member for member, time in ersvp[event_id] if member in member_set]))
        event_member_set = sorted(event_member_set)
        for i in range(len(event_member_set)):
            for j in range(i+1, len(event_member_set)):
                if (event_member_set[i], event_member_set[j]) in edges:
                    edges[(event_member_set[i], event_member_set[j])] += 1
                else:
                    edges[(event_member_set[i], event_member_set[j])] = 1
    member_set = set()
    for i,j in edges:
        member_set.add(i)
        member_set.add(j)
    return list(member_set), [(i, j, edges[(i, j)]) for i, j in edges]


def GetValidGrps(new_grp_event, groupjoin):
    return [i for i in new_grp_event if i in groupjoin]


def get_memberId_to_groups(valid_grps, groupjoin):
    memberId_to_groups = {}
    for grp in valid_grps:
        for member, time in groupjoin[grp]:
            if member in memberId_to_groups:
                memberId_to_groups[member].append(grp)
            else:
                memberId_to_groups[member] = [grp]
    return memberId_to_groups


def get_grp_to_valid_events(valid_grps, new_grp_event, ersvp):
    grp_to_valid_events = {}

    for i in valid_grps:
        grp_to_valid_events[i] = []
        for j, _ in new_grp_event[i]:
            if j in ersvp:
                if i in grp_to_valid_events:
                    grp_to_valid_events[i].append(j)
                else:
                    grp_to_valid_events[i] = [j]
    return grp_to_valid_events


def get_grp_to_memberId_to_attended_events(grp_to_valid_events, groupjoin, ersvp):
    grp_to_memberId_to_attended_events = {}
    for grp in grp_to_valid_events:
        memberId_to_attended_events = {}
        for event in grp_to_valid_events[grp]:
            for member, time in ersvp[event]:
                if member in memberId_to_attended_events:
                    memberId_to_attended_events[member].append(event)
                else:
                    memberId_to_attended_events[member] = [event]
        grp_to_memberId_to_attended_events[grp] = memberId_to_attended_events
    return grp_to_memberId_to_attended_events


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


def get_NMF(mat):
    model = NMF(n_components=6)
    model.fit(mat)
    return model.transform(mat)


def get_NMF_from_df(df):
    temp = df.copy()
    temp.drop(['member', 'group_id'], axis=1, inplace=True)
    temp = np.array(temp)
    nmf = get_NMF(temp)
    nmf = pd.DataFrame(nmf, columns=['r1', 'r2', 'r3', 'r4', 'r5', 'r6'])
    nmf["member"] = df["member"]
    nmf["group_id"] = df["group_id"]
    nmf = nmf[["member", "group_id", 'r1', 'r2', 'r3', 'r4', 'r5', 'r6']]
    return nmf


def make_member_level_features(grp_id, groupjoin, ersvp, new_grp_event, grp_to_memberId_to_attended_events, memberId_to_groups, grp_to_valid_events,window):
    members, edges = make_social_graph_of_grp_members(
        grp_id, groupjoin, ersvp, new_grp_event)
    if(len(members)==0) :
        return
    adjacency_list = {}
    for member in members:
        adjacency_list[member] = []
    for i, j, w in edges:
        adjacency_list[i].append((j, w))
        adjacency_list[j].append((i, w))
    member_level_features = {}
    member_level_features["member"] = []
    member_level_features["group_id"] = []
    member_level_features["m1"] = []
    member_level_features["m2"] = []
    member_level_features["m3"] = []
    member_level_features["m4"] = []
    member_level_features["m5"] = []
    member_level_features["m6"] = []
    member_level_features["m7"] = []
    member_level_features["m8"] = []
    member_level_features["m9"] = []
    member_level_features["m10"] = []
    member_level_features["m11"] = []
    member_level_features["m12"] = []
    member_to_index = {}
    ind = 0
    for member in members:
        member_to_index[member] = ind
        ind += 1
        member_level_features["member"].append(member)
        member_level_features["group_id"].append(grp_id)
        member_level_features["m1"].append(len(adjacency_list[member]))
        member_level_features["m2"].append(len(
            grp_to_memberId_to_attended_events[grp_id][member]) if member in grp_to_memberId_to_attended_events[grp_id] else 0)
        member_level_features["m3"].append(len(memberId_to_groups[member]))
        member_level_features["m4"].append(entropy([len(grp_to_memberId_to_attended_events[g_id][member])
                                           for g_id in memberId_to_groups[member] if member in grp_to_memberId_to_attended_events[g_id]]))
        member_level_features["m5"].append(entropy(
            [len(grp_to_valid_events[g_id]) for g_id in memberId_to_groups[member]]))
        member_level_features["m6"].append(entropy([len(grp_to_memberId_to_attended_events[g_id][member])/len(
            grp_to_valid_events[g_id]) for g_id in memberId_to_groups[member] if member in grp_to_memberId_to_attended_events[g_id] and len(
            grp_to_valid_events[g_id])>0]))
#     print(member_level_features)
    for member in members:
        if len(adjacency_list[member]):
            member_level_features["m7"].append(sum([member_level_features["m1"][member_to_index[neighbor[0]]]
                                               for neighbor in adjacency_list[member]])/len(adjacency_list[member]))
            member_level_features["m8"].append(sum([member_level_features["m2"][member_to_index[neighbor[0]]]
                                               for neighbor in adjacency_list[member]])/len(adjacency_list[member]))

            member_level_features["m9"].append(sum([member_level_features["m3"][member_to_index[neighbor[0]]]
                                               for neighbor in adjacency_list[member]])/len(adjacency_list[member]))
            member_level_features["m10"].append(sum([member_level_features["m4"][member_to_index[neighbor[0]]]
                                                for neighbor in adjacency_list[member]])/len(adjacency_list[member]))

            member_level_features["m11"].append(sum([member_level_features["m5"][member_to_index[neighbor[0]]]
                                                for neighbor in adjacency_list[member]])/len(adjacency_list[member]))
            member_level_features["m12"].append(sum([member_level_features["m6"][member_to_index[neighbor[0]]]
                                                for neighbor in adjacency_list[member]])/len(adjacency_list[member]))
        else:
            member_level_features["m7"].append(0)
            member_level_features["m8"].append(0)
            member_level_features["m9"].append(0)
            member_level_features["m10"].append(0)
            member_level_features["m11"].append(0)
            member_level_features["m12"].append(0)

    df = pd.DataFrame(member_level_features)
    df.to_csv("member_level_features_windowwise/"+str(grp_id) + "_" + str(window) + ".csv", index=False)
    nmf = get_NMF_from_df(df)
    nmf.to_csv("NMF_member_level_features_windowwise/" +str(grp_id) + "_" + str(window) + ".csv", index=False)
    graph_file = open("graph_windowwise/"+str(grp_id) +"_" + str(window) + ".txt","w")
    print(*members,file = graph_file)
    for i,j,w in edges :
        print(i, j, w, file=graph_file)


def main():
    groupjoin = pickle.load(open("data/groupjoin", "rb"))
    new_grp_event = pickle.load(open("data/new_grp_event", "rb"))
    ersvp = pickle.load(open("data/ersvp", "rb"))
    valid_grps = GetValidGrps(new_grp_event, groupjoin)
    tag_new_slide = pickle.load(open("data/tag_new_slide1", "rb"))
    valid_grps = set(valid_grps)
    valid_grps = set([i for i in valid_grps if i in tag_new_slide])
    window_size = 10

    memberId_to_groups = get_memberId_to_groups(valid_grps, groupjoin)

    grp_to_valid_events = get_grp_to_valid_events(
        valid_grps, new_grp_event, ersvp)
    grp_to_memberId_to_attended_events = get_grp_to_memberId_to_attended_events(
        grp_to_valid_events, groupjoin, ersvp)
    for grp_id in valid_grps:
        events = new_grp_event[grp_id]
        valid_events = set(grp_to_valid_events[grp_id])
        for window in range(len(events)-window_size+1):
            new_grp_event[grp_id] = events[window:window+10]
            grp_to_valid_events[grp_id] = [i for i,j in new_grp_event[grp_id] if i in valid_events]
            make_member_level_features(grp_id, groupjoin, ersvp, new_grp_event,
                                   grp_to_memberId_to_attended_events, memberId_to_groups, grp_to_valid_events,window)

        new_grp_event[grp_id] = events
        grp_to_valid_events[grp_id] = list(valid_events)



if __name__ == "__main__":
    main()
