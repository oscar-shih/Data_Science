import networkx as nx
from utility_funcs import readcirclefile, cost_function
import os
import numpy as np
import scipy as sp
import sklearn as sk
from sklearn import decomposition
def read_nodeadjlist(filename):
    ego = int(filename.rsplit('/', 1)[1].split('.')[0])
    G = nx.Graph()
    for line in open(filename):
        e1, es = line.split(':')
        G.add_edge(ego,int(e1))
        es = es.split()
        for e in es:
            if e == e1: 
                continue
            G.add_edge(int(e1),int(e))
    return G


def compute_training_score(cluster_function):
    trainingfiles = os.listdir('./data/Training/')
    pred_score = {}
    
    for item in trainingfiles:
        ego = int((item.split('.')[0]))
        true_circles = readcirclefile('./data/Training/'+item)
        G = read_nodeadjlist('./data/egonets/'+str(ego)+'.egonet')
        pred_circles = cluster_function(G)
        for key, val in pred_circles.items():
            if ego in val:
                pred_circles[key].remove(ego)
        pred_score[ego] = cost_function(pred_circles,true_circles)
    return pred_score

def NMF_cluster(G, beta=0.2, thresh=0.5, N_comp=20, min_circle=5):
    L = -nx.laplacian_matrix(G)
    expL = sp.linalg.expm(beta * L)
    
    N_nodes = len(G.nodes())
    B = np.zeros([N_nodes, N_nodes])
    
    for i in range(N_nodes):
        for j in range(N_nodes):
            B[i, j] = expL[i,j] / np.sqrt(expL[i, i] * expL[j, j])
            
    N_comp = min(N_comp,N_nodes)
            
    nmf = decomposition.NMF(n_components=N_comp)
    comps= nmf.fit_transform(B)

    pred_circles = dict(zip(range(N_comp),[[] for _ in range(N_comp)]))
    node_list = list(G.nodes())
    for i in range(N_comp):
        for j, val in enumerate(comps[:,i]):
            if val > thresh:
                pred_circles[i].append(node_list[j])
                print(pred_circles[i])
                
    return dict(((i,circle) for i, circle in pred_circles.items() if len(circle) >= min_circle))  

def writeline(_string,fh):
    fh.write(_string+'\n')

def write_test_file(output_file,cluster_function):
    test_egos = []
    with open('testSet_users_friends.csv','r') as f:
        for line in f:
            friend = line.split(',')[0]
            if friend != 'UserId':
                test_egos.append(int(line.split(',')[0]))
    with open(output_file,'w') as f:
        writeline('UserId,Predicted',f)
        for ego in test_egos:
            print('Processing...')
            G = read_nodeadjlist('./data/egonets/'+str(ego)+'.egonet')
            pred_circles = cluster_function(G)
            for key,val in pred_circles.items():
                if ego in val:
                    pred_circles[key].remove(ego)
            cs = [' '.join([str(y) for y in x]) for x in pred_circles.values() if len(x) > 0]
            outline =  str(ego) + ',' + ';'.join(cs)
            writeline(outline, f)    

if __name__ == "__main__":
    ego = 239
    G = read_nodeadjlist('./data/egonets/'+str(ego)+'.egonet')
    true_circles = readcirclefile('./data/Training/'+str(ego)+'.circles')

    pred_circle = NMF_cluster(G)
    print(ego, cost_function(pred_circle, true_circles))

    ego = 345
    G = read_nodeadjlist('./data/egonets/'+str(ego)+'.egonet')
    true_circles = readcirclefile('./data/Training/'+str(ego)+'.circles')

    pred_circle = NMF_cluster(G)
    print(ego, cost_function(pred_circle, true_circles))


    pred_score = compute_training_score(NMF_cluster)
    for ego, score in pred_score.items():
        print("Ego, Score: ", ego, score)
    print('Total: ', sum(pred_score.values()))
    # write_test_file('fixed_NMF.csv',NMF_cluster)
