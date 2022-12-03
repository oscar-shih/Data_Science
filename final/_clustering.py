import json
import os
from argparse import ArgumentParser
from pathlib import Path

import networkx as nx
from utility_funcs import read_circles, read_nodeadjlist, read_egonet, read_training_set
import os
import sklearn.cluster
import collections
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt


def plot_hist_and_box(data):
    plt.figure(plt.figsize(15, 5))
    plt.subplot(121)
    plt.hist(data)
    plt.subplot(122)
    plt.boxplot(data)


def main(args):
    G2 = read_egonet(args.eponet_file)
    print(len(nx.connected_component_subgraphs(G2)))
    egographs = dict()
    circles = dict()
    egodir = args.egonets_path
    circledir = args.training_path
    for circlefile in os.listdir(circledir):
        userID = os.path.splitext(circlefile)[0]
        egographs[int(userID)] = read_nodeadjlist(
            os.path.join(egodir, userID + ".egonet"))
        circles[int(userID)] = read_circles(
            os.path.join(circledir, circlefile))
    training_set = read_training_set(circledir, egodir)
    avg_ccs = [nx.average_clustering(egograph)
               for egograph in egographs.values()]
    plot_hist_and_box(avg_ccs)
    avg_cir_cc = list()
    for userID, circledict in circles.items():
        egograph = egographs[userID]
        for circleID, nodelist in circledict.items():
            circlenet = egograph.subgraph(nodelist)
            avg_cir_cc.append(nx.average_clustering(circlenet))
    plot_hist_and_box(avg_cir_cc)
    print(nx.clustering(egographs[args.userID]))
    egograph = egographs[args.userID]
    circle_dict = circles[args.user_ID]

    m = len(circle_dict.items()) + 1
    fig, axes = plt.subplots(m, 1, sharex=True, sharey=True)
    fig.set_size_inches(7, 3*m)

    data = nx.triangles(egograph).values()
    binlist = range(0, max(data) + 5, 5)

    axes[0].hist(data, bins=binlist)

    for n, (circle_ID, node_list) in enumerate(circle_dict.items()):
        axes[n+1].hist(nx.triangles(egograph.subgraph(node_list)
                                    ).values(), bins=binlist)
        plt.title(circle_ID)

    num_of_triangles = list()
    num_of_circles = list()
    for egograph, circledict in training_set.values():
        num_of_triangles.append(len(nx.triangles(egograph)))
        num_of_circles.append(len(circledict))
    print(num_of_triangles)
    plt.figure(plt.figsize(5, 5))
    plt.plot(num_of_triangles, num_of_circles, 'ro', alpha=0.5)

    userID = args.userID
    cluster_dicts = list()

    for cc in nx.connected_component_subgraphs(egographs[args.userID]):
        if len(cc.nodes()) > 8:
            cluster_labels = sklearn.cluster.spectral_clustering(
                nx.adjacency_matrix(cc))
            cluster_tuples = zip(cc.nodes(), cluster_labels)
            cluster_dict = collections.defaultdict(list)
            for userID2, clusterID in cluster_tuples:
                cluster_dict[clusterID].append(userID2)
            cluster_dicts.append(cluster_dict)

    output_string = str(userID) + ","
    circle_strings = list()
    for cluster_dict in cluster_dicts:
        for cluster_list in cluster_dict.values():
            circle_strings.append(' '.join(map(str, cluster_list)))
    output_string += ";".join(circle_strings)
    print(output_string)
    print(training_set.keys())
    print(list(nx.find_cliques(training_set['5881'][0])))
    print([(userID, len(x[0].nodes())) for userID, x in training_set.items()])
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
