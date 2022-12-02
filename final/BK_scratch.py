import os
import random
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import networkx as nx
import powerlaw

def readfeaturelist(filename):
    with open(filename) as f:
        out = []        # list of feature names
        for line in f:
            out.append(line.strip())
        return out

def readfeatures(featurefile):
    with open(featurefile) as f:
        out = [] 
        for line in f:
            tokens = line.split()
            profile = {}  # empty profile for the user
            for tok in tokens[1:]:
                feature,val = tok.rsplit(';',1)
                val = int(val)
                if feature not in profile:
                    profile[feature]=[val]
                else:
                    profile[feature].append(val)
            out.append(profile)
        for i in range(len(out)):
            assert out[i]['id'][0] == i  # check that each line was read and placed in the correct place in the list
        return out

def featurematch(profile1, profile2, feature):
    return len(set(profile1[feature]).intersection(set(profile2[feature]))) if feature in profile1 and feature in profile2 else 0

def matchvector(profile1, profile2, featurelist):
    out = []
    for feature in featurelist:
        out.append(featurematch(profile1, profile2, feature))
    return out

def weighteddotproduct(vector1, vector2, weight=None):
    if not weight:
        weight = np.ones(len(vector1))
    return np.inner(vector1, np.multiply(weight, vector2)) / np.mean(weight)

def userfeatures(profile):
    """  Returns a list of the features contained in the user profile """
    return [f for f in profile]

def usermatch(profile1, profile2):
    """ returns the match vector for profile2 using only profile1 features as a reference """
    return matchvector(profile1,profile2,userfeatures(profile1))

def readcircle(userID):
    """
    reads a circle for a given user consisting of circleDD: user1 user2 user3 ...
    and returns a dictionary of the circle['number']=[user1, user2, user3]
    """
    circlefile = './data/Training/'+str(userID)+'.circles'
    with open(circlefile) as f:
        circles = {} 
        for line in f:
            tokens = line.split()
            circleID = int(tokens[0].split('circle')[1].split(':')[0])
            circles[circleID] =[]
            for tok in tokens[1:]:
                circles[circleID].append(int(tok))
        return circles
def charprofile(profilemat):
    out = np.zeros(len(profilemat[0]))
    for row in profilemat:
        for i in range(len(row)):
            out[i] += row[i]
    return out
def display_char_profile(charprofile, featurelist):
    out = []
    for i in range(len(featurelist)):
        if charprofile[i] != 0: 
            out.append(str(featurelist[i]) + ": " + str(charprofile[i]))
    return out

def BK_scratch_outputfile(userid, ref_profile, circles, profile, features, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        print('User: ', userid, "\n", file=f)
        char_profile = {}
        for circle in circles:
            print('Circle:', circle, file=f)
            matchmatrix = [
                matchvector(ref_profile, profile[user], features) for user in circles[circle] 
            ]
            char_profile[circle] = charprofile(matchmatrix)
            out = display_char_profile(char_profile[circle], features)
            for char in out:
                print(char, file=f)
            print("\n", file=f)
def read_nodeadjlist(filename):
  G = nx.Graph()
  for line in open(filename):
    e1, es = line.split(':')
    # Add a node for the user.
    G.add_node(int(e1))
    es = es.split()
    for e in es:
      if e == e1: continue
      G.add_edge(int(e1),int(e))
  return G

def drawGraph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos)

if __name__ == '__main__':
    data_dir = "./data"
    features = readfeaturelist(os.path.join(data_dir, 'featureList.txt'))
    profile = readfeatures(os.path.join(data_dir, 'features.txt'))
    userid = 345
    out_path = "./BK_scratch.txt"
    ref_profile = profile[userid]
    circles = readcircle(userid)
    BK_scratch_outputfile(
        userid=userid,
        ref_profile=ref_profile,
        circles=circles,
        profile=profile,
        features=features,
        out_path=out_path
    ) # Generate file called "BK_scratch.txt"

    trainingfiles = os.listdir('./data/Training/')
    alltraining = {}
    for item in trainingfiles:
        ego = int((item.split('.')[0]))
        alltraining[ego] = readcircle(ego)
    circlesizes = []
    for ego in alltraining:
        for circle in alltraining[ego]:
            circlesizes.append(len(alltraining[ego][circle]))

    n, bins, patches = plt.hist(circlesizes, 50)
    fit = powerlaw.Fit(circlesizes, xmin=1.0)
    for fitname in fit.supported_distributions:
        if fit.supported_distributions[fitname]: 
            print(fitname, fit.distribution_compare('power_law', fitname ,normalized_ratio=True))
    fig = fit.plot_pdf(color='b', linewidth=2)
    fit.power_law.plot_pdf(color='b', linestyle='--', ax=fig)
    fit.truncated_power_law.plot_pdf(color='r', linestyle='--', ax=fig)
    fig.figure.savefig("test.png")

    ref_user=239
    ref_circle = alltraining[ref_user]
    ref_circle_nums = [c for c in ref_circle]
    # ref_circle_nums = ref_circle.keys()

    print('User:',ref_user,'\nCircle #\'s:', ref_circle_nums)
    G = read_nodeadjlist('./data/egonets/'+str(ref_user)+'.egonet')
    pos = nx.spring_layout(G)
    plt.figure(figsize=(15, 15))
    for (n, c) in enumerate(ref_circle_nums):
        plt.subplot(2, 2, n+1)
        nx.draw(G, pos, node_color='r', node_size=5)  # draw the background graph as red
        nx.draw(G.subgraph(ref_circle[c]), pos, node_color='b', node_size=5)  # draw each subgraph as blue
        plt.title(f"Egonet of User{ref_user}")
    plt.savefig(f"final.png")  # plot each as a separate plot
    clust = nx.clustering(G)
    fig = plt.figure(figsize=(8,3))
    plt.hist(clust.values(), bins=20)
    plt.savefig("Histogram.png")

    friendly_users = [user_id for user_id, coeff in clust.items() if coeff == 1]
    fig = plt.figure(figsize=(5,5))
    plt.title("Friendly_users")
    nx.draw(G, pos, node_color='r', node_size=5)  # draw the background graph as red
    nx.draw(G.subgraph(friendly_users),pos,node_color='b', node_size=5)
    # plt.show()


    ref_user = 345
    ref_circle = alltraining[ref_user]
    ref_circle_nums = [c for c in ref_circle]
    G = read_nodeadjlist('./data/egonets/'+str(ref_user)+'.egonet')
    pos = nx.spring_layout(G) # positions for all nodes

    ncolormap = {}
    fig = plt.figure(figsize=(20,20))
    for c in ref_circle:
        ncolormap[c] = (random.random(),random.random(),random.random())
        print(ncolormap[c])
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=ref_circle[c],
            node_color=ncolormap[c],
            node_size=15,
            alpha=0.8
        )
    #edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    plt.savefig("Final.png")
    plt.show()




