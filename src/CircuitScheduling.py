import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import copy
import random


def max_degree(graph):
    return max(list(dict(graph.degree).values()))

def BipartitieGraphFromCheckMat(H):
    num_checks, num_bits = H.shape
    C_nodes = list(-np.arange(1, num_checks + 1))
    V_nodes = list(np.arange(1, num_bits + 1))
    edges = [(-(i + 1), j + 1) for i in range(num_checks) for j in range(num_bits) if H[i][j] == 1]
    
    G = nx.Graph()
    G.add_nodes_from(C_nodes, bipartite=0)
    G.add_nodes_from(V_nodes, bipartite=1)
    G.add_edges_from(edges)
    
    return G

def best_match(graph):
    C_nodes = list({n for n, d in graph.nodes(data=True) if d["bipartite"] == 0})
    V_nodes = list(set(graph) - set(C_nodes))

    return bipartite.matching.hopcroft_karp_matching(graph, C_nodes)

# Coloration circuit
def TransformBipartiteGraph(G):
    # transform any bipartite graph to a symmetric one by adding dummy vertices and edges
    G_s = copy.deepcopy(G)
    C_nodes = list({n for n, d in G.nodes(data=True) if d["bipartite"] == 0})
    V_nodes = list(set(G) - set(C_nodes))
    
    # Suppose C_nodes all have degree Delta_c, and # V_nodes > # C_nodes
    # Add dummy vertices to C_nodes
    C_nodes_dummy = list(-np.arange((len(C_nodes) + 1), len(V_nodes) + 1))
    G_s.add_nodes_from(C_nodes_dummy, bipartite=0)
    
    # Add dummy edges between edges with degree < Delta_c
    Delta = max_degree(G_s)
#     print('max degree:', Delta)
    open_degree_nodes = copy.deepcopy(dict((node, degree) for node, degree in dict(G_s.degree()).items() if degree < Delta))
            
    while len(open_degree_nodes) > 0:       
        for node1 in list(open_degree_nodes.keys()):
            if node1 < 0:
                c_node = node1
                for node2 in list(open_degree_nodes.keys()):
                    if node2 > 0:
                        v_node = node2
                        if not G_s.has_edge(c_node, v_node):
                            G_s.add_edge(c_node, v_node)
                            
                            if open_degree_nodes[c_node] + 1 == Delta:
                                open_degree_nodes.pop(c_node)
                            else:
                                open_degree_nodes[c_node] = open_degree_nodes[c_node] + 1

                            if open_degree_nodes[v_node] + 1 == Delta:
                                open_degree_nodes.pop(v_node)
                            else:
                                open_degree_nodes[v_node] = open_degree_nodes[v_node] + 1
                            
                            break            
                        
            
    return G_s


def edge_corloring(graph):
    matches_list = []
    g = copy.deepcopy(graph)
    g_s = TransformBipartiteGraph(g)
    
    number_colors = max_degree(g_s)
    
    C_nodes = list({n for n, d in g.nodes(data=True) if d["bipartite"] == 0})
    V_nodes = list(set(g) - set(C_nodes))
    
    C_nodes_s = list({n for n, d in g_s.nodes(data=True) if d["bipartite"] == 0})
    V_nodes_s = list(set(g_s) - set(C_nodes_s))

    while len(g_s.edges()) > 0:
#         print('NEXT COLOR')
        bm=best_match(g_s)
#         print(bm)
#         matches_list.append(bm)
        
        # find the uniqe edges
        unique_match = dict((c_node, bm[c_node]) for c_node in bm if c_node in C_nodes)
        edges_list = [(c_node, bm[c_node]) for c_node in bm if c_node in C_nodes_s]
        matches_list.append(unique_match)
        
        g_s.remove_edges_from(edges_list)
#     assert len(g.edges()) == 0
        
    return matches_list

def ColorationCircuit(H):
    G = BipartitieGraphFromCheckMat(H)
    matches_list = edge_corloring(G)
    
    scheduling_list = []
    for match in matches_list:
        scheduling_list.append(dict((-c_node - 1, match[c_node] - 1) for c_node in match))
    
    return scheduling_list




# Random circuit
def RandomCircuit(H):
    # Obtain a random scheduling 
    rand_scheduling_seed = 30000
    num_checks, num_bits = H.shape
    max_stab_w = max([int(np.sum(H[i,:])) for i in range(num_checks)])
    scheduling_list = [list(np.where(H[ancilla_index,:] == 1)[0]) for ancilla_index in range(num_checks)]
    [random.Random(i + rand_scheduling_seed).shuffle(scheduling_list[i]) for i in range(len(scheduling_list))]
    
    schedulings = []
    for time_step in range(max_stab_w):
        scheduling = {}
        for ancilla_index in range(num_checks):
            if len(scheduling_list[ancilla_index]) >= time_step + 1:
                scheduling[ancilla_index] = scheduling_list[ancilla_index][time_step]
        schedulings.append(scheduling)
    return schedulings