import json
import os
from argparse import ArgumentParser
from pathlib import Path

import networkx as nx
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
from connected_components import read_nodeadjlist, findCommunities

def drawGraph(G):
    pos = nx.shell_layout(G)
    nx.draw(G, pos)
    
def common_friends(args):
    egofile = open(args.eponet_file)
    result = dict()
    for line in egofile:
        v_i, v = line.split(":")
        result[v_i] = len(v.split())
    return result

def circle_sizes(circle_file_path):
    "Returns a dict, indexed by circle_id, of each circle size."
    circle_file = open(circle_file_path)
    result = dict()
    for line in circle_file:
        circle_id, friends = line.split(':')
        result[circle_id] = len(friends.split())
    return result

def display_dist_with_circles(user_id, bins=20):
    plt.hist(common_friends('egonets/' + str(user_id) + '.egonet').values(), bins=bins)
    plt.title("User " + str(user_id))
    for circle_size in sorted(circle_sizes('Training/' + str(user_id) + '.circles').values()):
        plt.axvline(x=circle_size, color='r')

def get_friend_count_matrix(egofile_path):
    """Return a dict of dicts giving a matrix (indexed by ID) of friends in common.
    Should be a symmetric matrix with the values of common_friends(egofile_path) as
    the diagonal elements."""
    
    egofile = open(egofile_path)
    friend_sets = dict()
    for line in egofile:
        v_i, v = line.split(":")
        friend_sets[v_i] = set(v.split())
    result = dict()
    for v_i in friend_sets.keys():
        result[v_i] = dict()
        for v_j in friend_sets.keys():
            result[v_i][v_j] = len(friend_sets[v_i].intersection(friend_sets[v_j]))
    return result

def main(args):
    G = read_nodeadjlist(args.eponet_file)
    G2 = nx.subgraph(G, G.neighbors(87))
    drawGraph(G2)
    
    ccs = dict()
    for friend in G.nodes():
        ffriends = G.neighbors(friend)
        subgraph = nx.subgraph(G, ffriends)
        fcc = nx.connected_components(subgraph)
        ccs[friend] = len(fcc)
    print(ccs)
    
    G = read_nodeadjlist(args.eponet_file)
    common_friends = []
    for friend in G.nodes():
        common_friends.append(len(G.neighbors(friend)))
    plt.hist(common_friends)
    
    # print(nx.triangles(G))
    
    # cf = common_friends(args.eponet_file)
    # plt.hist(cf.values())
    
    # print(sum([x == 0 for x in cf.values()]))
    # print(circle_sizes('Training/239.circles'))
    # display_dist_with_circles(239)
    # display_dist_with_circles(345)
    
    # circle_files = os.listdir('Training')
    # plt.figure(plt.figsize(18,60))
    # for (n, circle_file) in enumerate(circle_files):
    #     user_id = int(os.path.splitext(circle_file)[0])
    #     plt.subplot(20, 3, n+1)
    #     display_dist_with_circles(user_id)
    
    # ccf239 = get_friend_count_matrix('egonets/239.egonet')
    # assert len(ccf239) == len(cf239)
    # for i in ccf239.keys():
    #     for j in ccf239[i].keys():
    #         assert ccf239[i][j] == ccf239[j][i]
    # for i in ccf239.keys():
    #     assert ccf239[i][i] == cf239[i]
        
    # plt.figure(figsize=(3,5))
    # values = []
    # for v_i, x in ccf239.items():
    #     for v_j, n in x.items():
    #         if int(v_i) > int(v_j):
    #             values.append(int(n))
    # plt.ylim(0,600)
    # plt.hist(values)
    
        
    egofile = open(args.eponet_file)
    friend_sets = dict()
    for line in egofile:
        v_i, v = line.split(":")
        friend_sets[v_i] = set(v.split())
    result = dict()
    for v_i in friend_sets.keys():
        result[v_i] = dict()
        for v_j in friend_sets.keys():
            result[v_i][v_j] = len(friend_sets[v_i].intersection(friend_sets[v_j]))
    print(result)
    values = []
    for v_i, x in result.items():
        for v_j, n in x.items():
            if int(v_i) > int(v_j):
                values.append(int(n))
    plt.ylim(0, 600)
    plt.hist(values, bins=30)
    pass

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--featureList", type=Path,
                        default="./featureList.txt")
    parser.add_argument("--training_path", type=Path,
                        default="./Training")
    parser.add_argument("--egonets_path", type=Path,
                        default="./egonets")
    parser.add_argument("--features", type=Path,
                        default="./features.txt")
    parser.add_argument("--eponet_file", type=Path,
                        default="egonets/0.egonet")
    parser.add_argument("--output_csv", type=Path,
                        default="characterist_profiles.csv")
    parser.add_argument("--userID", type=int,
                        default=239)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
