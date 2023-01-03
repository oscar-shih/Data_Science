import networkx as nx
from utility_funcs import readcirclefile, cost_function
import os
import sklearn.cluster
import community
import itertools
import matplotlib.pyplot as plt

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

def writeline(_string, fh):
    fh.write(_string + '\n')

def write_test_file(output_file, cluster_function):
    test_egos = []
    with open('sample_submission.csv','r') as f:
        for line in f:
            friend = line.split(',')[0]
            if friend != 'UserId':
                test_egos.append(int(line.split(',')[0]))
    with open(output_file,'w') as f:
        writeline('UserId, Predicted', f)
        for ego in test_egos:
            print('Processing...')
            G = read_nodeadjlist('./data/egonets/' + str(ego) + '.egonet')
            pred_circles = cluster_function(G)
            for key,val in pred_circles.items():
                if ego in val:
                    pred_circles[key].remove(ego)
            cs = [' '.join([str(y) for y in x]) for x in pred_circles.values() if len(x) > 0]
            outline =  str(ego) + ',' + ';'.join(cs)
            writeline(outline, f)   

def naive_spec_cluster(G,k=8):
    cl = sklearn.cluster.spectral_clustering(nx.adjacency_matrix(G), n_clusters=k)
    pred_circles = {}
    for circle,user in zip(cl,G.nodes()):  # ordering is the same as G.nodes()
        if circle in pred_circles:
            pred_circles[circle].append(user)
        else:
            pred_circles[circle]=[user]
    return pred_circles

def compute_training_score(cluster_function):
    trainingfiles = os.listdir('./data/Training/')
    pred_score = {}
    
    for item in trainingfiles:
        ego = int((item.split('.')[0]))
        true_circles = readcirclefile('./data/Training/'+item)
        G = read_nodeadjlist('./data/egonets/'+str(ego)+'.egonet')
        pred_circles = cluster_function(G)
        for key,val in pred_circles.items():
            if ego in val:
                pred_circles[key].remove(ego)
        pred_score[ego] = cost_function(pred_circles,true_circles)
    return pred_score

def dendrogram_cluster(G):
    part = community.best_partition(G)
    circles = dict()
    for user,label in part.items():
        if label in circles:
            circles[label].append(user)
        else:
            circles[label] = [user]
    return circles

def convert_circles_to_partition(circles):
    return dict(list(itertools.chain(*[[(user,label) for user in circle] for label, circle in circles.items()])))

def dynamic_spec_cluster(G,max_K=15):
    Adj_mat = nx.adjacency_matrix(G)
    max_K = min(max_K,len(G.nodes()))
    max_mod = -2
    for k in range(1,max_K):
        cl = sklearn.cluster.spectral_clustering(Adj_mat, n_clusters=k)
        pred_circles = {}
        for circle,user in zip(cl,G.nodes()):  # ordering is the same as G.nodes()
            if circle in pred_circles:
                pred_circles[circle].append(user)
            else:
                pred_circles[circle]=[user]
        pred_part = convert_circles_to_partition(pred_circles)
        pred_mod = community.modularity(pred_part,G)
        if pred_mod > max_mod:
            out = pred_circles
            max_mod = pred_mod
    return out

if __name__ == "__main__":
    cl_test = compute_training_score(naive_spec_cluster)
    # for ego, score in cl_test.items():
    #     print("Ego, Score: ", ego, score)
    print('Total: ', sum(cl_test.values()))
    # write_test_file('fully_conn_naive_spec_no_ego.csv', naive_spec_cluster)
    # # Test with ego = 345
    # ego = 345
    # G = read_nodeadjlist('./data/egonets/'+str(ego)+'.egonet')
    # pred_circles = naive_spec_cluster(G)
    # part = community.best_partition(G)
    # community.modularity(part, G)
    # spec_scores = compute_training_score(naive_spec_cluster)
    # dend_scores = compute_training_score(dendrogram_cluster)

    # write_test_file('dendrogram_no_ego.csv', dendrogram_cluster)
    # # Test with ego = 239
    # ego = 239
    # G = read_nodeadjlist('./data/egonets/'+str(ego)+'.egonet')
    # modular_K = []
    # for K in range(1,19):
    #     pred_circles = naive_spec_cluster(G,k=K)
    #     pred_part = convert_circles_to_partition(pred_circles)
    #     modular_K.append(community.modularity(pred_part,G))
    #     print(K, modular_K[K-1])
    # plt.plot(range(1,19), modular_K)
    # plt.show()

    dyn_spec_scores = compute_training_score(dynamic_spec_cluster)
    # for ego, score in dyn_spec_scores.items():
    #     print("Ego, Score: ", ego, score)
    print('Total: ', sum(dyn_spec_scores.values()))
    # G = read_nodeadjlist('./data/egonets/'+str(ego)+'.egonet')
    # true_circles = readcirclefile('./data/Training/'+str(ego)+'.circles')
    # dyn_test = dynamic_spec_cluster(G)
    # naive_test = naive_spec_cluster(G)
    # write_test_file('dynamic_spec_no_ego.csv', dynamic_spec_cluster)