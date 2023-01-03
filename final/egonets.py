import networkx as nx
from collections import defaultdict
import sys

def read_nodeadjlist(filename):
  G = nx.Graph()
  for line in open(filename):
    e1, es = line.split(':')
    es = es.split()
    for e in es:
      if e == e1: continue
      G.add_edge(int(e1),int(e))
  return G

def findCommunities(filename):
  G = read_nodeadjlist(filename)
  c = nx.connected_components(G)
  return c


G = read_nodeadjlist("./egonets/0.egonet")


def drawGraph(G):
    pos = nx.shell_layout(G)
    nx.draw(G, pos)

G2 = nx.subgraph(G, G.neighbors(87))


drawGraph(G2)



ccs = dict()
for friend in G.nodes():
    ffriends = G.neighbors(friend)
    subgraph = nx.subgraph(G, ffriends)
    fcc = nx.connected_components(subgraph)
    ccs[friend] = len(fcc)
ccs


G = read_nodeadjlist("0.egonet")
common_friends = []
for friend in G.nodes():
    common_friends.append(len(G.neighbors(friend)))
hist(common_friends)


def common_friends(egofile):
    egofile = open("0.egonet")
    result = dict()
    for line in egofile:
        v_i, v = line.split(":")
        result[v_i] = len(v.split())
    return result


cf = common_friends("0.egonet")


hist(cf.values())



sum([ x == 0 for x in cf.values()])

egofile = open("0.egonet")
friend_sets = dict()
for line in egofile:
    v_i, v = line.split(":")
    friend_sets[v_i] = set(v.split())
result = dict()
for v_i in friend_sets.keys():
    result[v_i] = dict()
    for v_j in friend_sets.keys():
        result[v_i][v_j] = len(friend_sets[v_i].intersection(friend_sets[v_j]))
result


values = []
for v_i, x in result.items():
    for v_j, n in x.items():
        if int(v_i) > int(v_j):
            values.append(int(n))
ylim(0,600)
hist(values, bins=30)


