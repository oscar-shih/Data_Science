import json
import os
from argparse import ArgumentParser
from pathlib import Path

import networkx as nx
from utility_funcs import readcirclefile, read_nodeadjlist, cost_function, readfeatures, readfeaturelist
import os
import sklearn.cluster
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt


def convert_profile_dict_to_vector(profile, features):
    out = []
    for feature in features:
        if feature in profile:
            out.append(profile[feature])
        else:
            out.append(set())
    return out


def match_vector(profile1, profile2):
    return [len(x.intersection(y)) for x, y in zip(profile1, profile2)]


def generate_feature_matrix(profiles_dict, ego, G):
    return [match_vector(profiles_dict[ego], profiles_dict[g]) for g in G.nodes()]


def generate_class_matrix(G, true_circles):
    return dict(zip(true_circles.keys(), [[int(g in circle) for g in G.nodes()] for circle in true_circles.values()]))


def main(args):
    features = readfeaturelist(args.featureList)
    print(features, len(features))
    profiles_dict = readfeatures(args.featureList)
    profile_matrix = [convert_profile_dict_to_vector(
        profile, features) for profile in profiles_dict]
    ego = args.ego
    circle_file = str(ego)+'.circles'
    egonet_file = str(ego)+'.egonet'
    true_circles = readcirclefile(
        os.path.join(args.training_path, circle_file))
    G = read_nodeadjlist(os.path.join(args.egonets_path, egonet_file))
    print('Total friends:', len(G.nodes()))
    class_matrix = generate_class_matrix(G, true_circles)
    feature_matrix = generate_feature_matrix(profile_matrix, ego, G)

    for label, circle in class_matrix.items():
        print('Training Ego:', ego, 'Circle:', label, '...')
        forest = RandomForestClassifier(n_estimators=100)
        forest = forest.fit(feature_matrix, circle)
        important_features = sorted(
            zip(features, forest.feature_importances_), key=lambda x: x[1], reverse=True)
        importance_scores = [val for key, val in important_features]
        importance_labels = [key for key, val in important_features]
        ind = range(len(importance_scores))
        plt.bar(ind, forest.feature_importances_)
        plt.axis([min(ind), max(ind), 0, 0.7])
        plt.show()
    dict(zip(true_circles.keys(), [
         [int(g in circle) for g in G.nodes()] for circle in true_circles.values()]))
    t = [[1, 2, 3], [4, 5, 6], [2, 6, 1]]
    l = ['a', 'b', 'c']
    dict(zip(l, t))
    true_circles = readcirclefile(
        os.path.join(args.training_path, circle_file))
    G = read_nodeadjlist(os.path.join(args.egonets_path, egonet_file))
    print('Total friends:', len(G.nodes()))
    class_matrix = generate_class_matrix(G, true_circles)
    feature_matrix = generate_feature_matrix(profile_matrix, ego, G)

    for label, circle in class_matrix.items():
        print('Training Ego:', ego, 'Circle:', label, '...')
        forest = RandomForestClassifier(n_estimators=100)
        forest = forest.fit(feature_matrix, circle)
        important_features = sorted(
            zip(features, forest.feature_importances_), key=lambda x: x[1], reverse=True)
        importance_scores = [val for key, val in important_features]
        importance_labels = [key for key, val in important_features]
        ind = range(len(importance_scores))
        plt.bar(ind, forest.feature_importances_)
        plt.axis([min(ind), max(ind), 0, 0.7])
        plt.show()

    trainingfiles = os.listdir(args.training_path)
    df_labels = ['Ego', 'Circle']+features
    characteristic_profiles = []

    for item in trainingfiles:
        ego = int((item.split('.')[0]))
        true_circles = readcirclefile(os.path.join(args.training_path, item))
        G = read_nodeadjlist(os.path.join(args.egonets_path, egonet_file))
        class_matrix = generate_class_matrix(G, true_circles)
        feature_matrix = generate_feature_matrix(profile_matrix, ego, G)

        for label, circle in class_matrix.items():
            print('Training Ego:', ego, 'Circle:', label, '...')
            forest = RandomForestClassifier(n_estimators=100)
            forest = forest.fit(feature_matrix, circle)
            characteristic_profiles.append(
                [ego]+[label]+list(forest.feature_importances_))

    df = pd.DataFrame(data=characteristic_profiles, columns=df_labels)
    print(df)
    print(df.mean())
    df.to_csv(args.output_csv)
    df_pos = df[df.min(axis=1) >= 0]
    df_neg = df[df.min(axis=1) < 0]
    df_pos_mean = df_pos.mean()
    df_pos_mean.sort(ascending=False)
    rand_chance = (len(df_pos_mean)-2)
    print(df_pos_mean[df_pos_mean.gt(1./(rand_chance))]*rand_chance)
    print(df_pos_mean*rand_chance)
    print(df_pos.mean()[2:])

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
    parser.add_argument("--output_csv", type=Path,
                        default="characterist_profiles.csv")
    parser.add_argument("--ego", type=int,
                        default=345)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
