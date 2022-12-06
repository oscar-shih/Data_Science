import os
import random
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import networkx as nx
import powerlaw

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

def read_nodeadjlist(filename):
    G = nx.Graph()
    ek = []
    for line in open(filename):
        e1, es = line.split(':')
        ek.append(e1)
        # Add a node for the user.
        G.add_node(int(e1))
        es = es.split()
        for e in es:
            if e == e1: 
                continue
            G.add_edge(int(e1),int(e))
    return G, ek

def drawGraph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos)

if __name__ == '__main__':
    trainingfiles = os.listdir('./data/Training/')
    alltraining = {}
    for item in trainingfiles:
        ego = int((item.split('.')[0]))
        alltraining[ego] = readcircle(ego)
    circlesizes = []
    for ego in alltraining:
        for circle in alltraining[ego]:
            circlesizes.append(len(alltraining[ego][circle]))

    ref_user = 345
    ref_circle = alltraining[ref_user]
    G, e = read_nodeadjlist('./data/egonets/'+str(ref_user)+'.egonet')
    pos = nx.spring_layout(G, k=0.5, center=(0,0), seed=1126) # positions for all nodes
    ncolormap = {}
    users = {}
    fig = plt.figure(figsize=(20,20))
    for c in ref_circle:
        ncolormap[c] = (random.random(),random.random(),random.random())
        # for user in ref_circle[c]:
        #     users[user] = int(user)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=ref_circle[c],
            node_color=ncolormap[c],
            node_size=55,
            alpha=0.8,
            label='circle' + str(c),
        )
        # nx.draw_networkx_labels(G, pos, users)
    #edges
    nx.draw_networkx_edges(G, pos, width=0.1, alpha=0.8)
    plt.legend(loc='best')
    plt.savefig("graph.png")
    plt.show()

    ref_circle_nums = [c for c in ref_circle]
    
    print('User:',ref_user,'\nCircle #\'s:', ref_circle_nums)
    color = (0.9459745640876799, 0.735602886359877, 0.36705459898363024)
    sub_color = (0.4402389827645813, 0.8926765960561694, 0.25484969751025355)
    for c in ref_circle_nums:
        labels = {}
        print('User:',ref_user,'Circle:',c,'Size:',len(ref_circle[c]))
        print('Members:', ref_circle[c])
        for user in ref_circle[c]:
            labels[user] = user
        fig = plt.figure(figsize=(15,15))
        nx.draw(G, pos, node_color=color, node_size=100, width=0.2)  # draw the background graph as red
        nx.draw(G.subgraph(ref_circle[c]), pos, node_color=sub_color, node_size=100, width=0.2)  # draw each subgraph as blue
        nx.draw_networkx_labels(G, pos, labels, font_color='black', font_size=6)
        plt.savefig(f"Circle_{c}.png")
        plt.show()  # plot each as a separate plot


    clust = nx.clustering(G)
    plt.figure(figsize=(8,3))
    plt.hist(clust.values(), bins=20)
    plt.savefig("Histogram.png")
    plt.show()

    friendly_users = [user_id for user_id, coeff in clust.items() if coeff == 1.0]
    labels = {}
    print("Friendly Users are: ", friendly_users)
    for user in friendly_users:
        labels[user] = user
    plt.figure(figsize=(15,15))
    nx.draw(G, pos, node_color=color, node_size=55, width=0.2)  # draw the background graph as red
    nx.draw(G.subgraph(friendly_users), pos, node_color=sub_color, node_size=55, width=0.2)  # draw each subgraph as blue
    nx.draw_networkx_labels(G, pos, labels)
    plt.savefig("friend.png")
    plt.show() 