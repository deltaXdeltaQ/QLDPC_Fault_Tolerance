import numpy as np
import matplotlib.pyplot as plt
from graph_tools import Graph
import networkx as nx
import random
import copy
import time
import json

import ldpc
import bposd

from bposd.css_decode_sim import css_decode_sim
from bposd.hgp import hgp
import pickle

from scipy.optimize import curve_fit

from ldpc.code_util import compute_code_distance
import ldpc.mod2 as mod2

from itertools import combinations
from itertools import permutations


def Girth(G):
#     cycles = [len(cycle) for cycle in nx.cycle_basis(G)]
    return min([len(cycle) for cycle in nx.cycle_basis(G)]) if nx.cycle_basis(G) != [] else 1e7

def QuantumExpanderFromCheckMat(H):
    h1=H
    h2=H
    qcode=hgp(h1,h2,compute_distance=True)
    return qcode
    
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
def load_object(filename):
    with open(filename, 'rb') as inp:
        return pickle.load(inp)

def TannerGraphToCheckMat(G):
    C_nodes = list({n for n, d in G.nodes(data=True) if d["bipartite"] == 0})
    V_nodes = list(set(G) - set(C_nodes))
    
#     print('C_nodes:', C_nodes, 'V_nodes:', V_nodes)
    edge_list = list(G.edges())
    
#     n_checks = n0*Delta_v
#     n_bits = n0*Delta_c
    n_checks = len(C_nodes)
    n_bits = len(V_nodes)
    p_mat = np.zeros([n_checks, n_bits])

    for edge in edge_list:
        if edge[0] < 0:
            edge = (edge[1], edge[0])
        c_index, v_index = C_nodes.index(edge[0]), V_nodes.index(edge[1])
        p_mat[c_index, v_index] = 1
        
    return p_mat

def GetClassicalCodeParams(H):
    n=H.shape[1] #the code block length
    k=n-mod2.rank(H) #This follows from the rank-nullity theorem
    d=compute_code_distance(H) #This function exhaustively computes the code distance by checking the weight of all logical operators (Not scalable!).
    
    eval_mat = np.matmul(np.transpose(H), H)
    lambda_2 = np.linalg.eigvals(eval_mat)[1]
    
    return [n, k, d, lambda_2]


def DSwitch(input_pairs):
    pairs = copy.deepcopy(input_pairs)
    pairs_list = list(pairs.keys())
    multiplicity_list = list(pairs.values())
    
    while 2 in multiplicity_list:
        double_pair_index = multiplicity_list.index(2)
        double_pair = pairs_list[double_pair_index]
        # randomly draw two single pairs
        single_pairs_indices = np.where(np.array(multiplicity_list) == 1)[0]
        
        rand_single_pairs = []
        for i in range(2):
            c_nodes_list = [double_pair[0]]
            v_nodes_list = [double_pair[1]]
            for rand_single_pair in rand_single_pairs:
                c_nodes_list.append(rand_single_pair[0])
                v_nodes_list.append(rand_single_pair[1])               
       
            if_success = False
            while not if_success:
                rand_single_pair = pairs_list[np.random.choice(single_pairs_indices, size = 1, replace = False)[0]]
                if (rand_single_pair not in rand_single_pairs) and \
                    (rand_single_pair[0] not in c_nodes_list) and \
                    (rand_single_pair[1] not in v_nodes_list) and \
                    ((double_pair[0], rand_single_pair[1]) not in pairs_list) and \
                    ((rand_single_pair[0], double_pair[1]) not in pairs_list):
                    if_success = True
                    rand_single_pairs.append(rand_single_pair)
        
#         print('Swap:', 'doubled pair:', double_pair, 'single pairs:', rand_single_pairs)
        ## Remove old pairs
        pairs.pop(double_pair)
        for single_pair in rand_single_pairs:
            pairs.pop(single_pair)
        
        ## Add new pairs   
        for i in range(len(rand_single_pairs)):
            if (double_pair[0], rand_single_pairs[i][1]) in list(pairs.keys()):
                pairs[(double_pair[0], rand_single_pairs[i][1])] += 1
            else:
                pairs[(double_pair[0], rand_single_pairs[i][1])] = 1
                
            if (rand_single_pairs[i][0], double_pair[1]) in list(pairs.keys()):
                pairs[(rand_single_pairs[i][0], double_pair[1])] += 1
            else:
                pairs[(rand_single_pairs[i][0], double_pair[1])] = 1
        
        pairs_list = list(pairs.keys())
        multiplicity_list = list(pairs.values())
    
    return pairs
        
def TSwitch(input_pairs):
    pairs = copy.deepcopy(input_pairs)
    pairs_list = list(pairs.keys())
    multiplicity_list = list(pairs.values())
    
    while 3 in multiplicity_list:
        triple_pair_index = multiplicity_list.index(3)
        triple_pair = pairs_list[triple_pair_index]
        # randomly draw three single pairs 
        single_pairs_indices = np.where(np.array(multiplicity_list) == 1)[0]
        rand_single_pairs = []
        for i in range(3):
            c_nodes_list = [triple_pair[0]]
            v_nodes_list = [triple_pair[1]]
            for rand_single_pair in rand_single_pairs:
                c_nodes_list.append(rand_single_pair[0])
                v_nodes_list.append(rand_single_pair[1])               
       
            if_success = False
            while not if_success:
                rand_single_pair = pairs_list[np.random.choice(single_pairs_indices, size = 1, replace = False)[0]]
                if (rand_single_pair not in rand_single_pairs) and \
                    (rand_single_pair[0] not in c_nodes_list) and \
                    (rand_single_pair[1] not in v_nodes_list) and \
                    ((triple_pair[0], rand_single_pair[1]) not in pairs_list) and \
                    ((rand_single_pair[0], triple_pair[1]) not in pairs_list):
                    if_success = True
                    rand_single_pairs.append(rand_single_pair)
                    
        ## Remove old pairs
        pairs.pop(triple_pair)
        for single_pair in rand_single_pairs:
            pairs.pop(single_pair)
        
        ## Add new pairs
        for i in range(len(rand_single_pairs)):
            if (triple_pair[0], rand_single_pairs[i][1]) in list(pairs.keys()):
                pairs[(triple_pair[0], rand_single_pairs[i][1])] += 1
            else:
                pairs[(triple_pair[0], rand_single_pairs[i][1])] = 1
                
            if (rand_single_pairs[i][0], triple_pair[1]) in list(pairs.keys()):
                pairs[(rand_single_pairs[i][0], triple_pair[1])] += 1
            else:
                pairs[(rand_single_pairs[i][0], triple_pair[1])] = 1
                
        pairs_list = list(pairs.keys())
        multiplicity_list = list(pairs.values())
    
    return pairs


def RandomaGraphs(n0:int, Delta_c:int, Delta_v:int):
    ## G: empty graph, Delta_c (Delta_v): degree of check (variable) nodes
    num_check_nodes = n0*(Delta_v)
    num_variable_nodes = n0*(Delta_c)
    
    G=nx.Graph()
    G.add_nodes_from(list(np.arange(1, num_check_nodes + 1)), bipartite=0) # Check nodes
    G.add_nodes_from(list(- (np.arange(1,num_variable_nodes + 1))), bipartite=1) # Variable nodes

    C_nodes = list({n for n, d in G.nodes(data=True) if d["bipartite"] == 0})
    V_nodes = list(set(G) - set(C_nodes))

    C_ports = []
    for c_node in C_nodes:
        C_ports += [c_node]*Delta_c
    random.shuffle(C_ports)
    
    V_ports = []
    for v_node in V_nodes:
        V_ports += [v_node]*Delta_v
    random.shuffle(V_ports)
    
    pairs = {}
    for i in range(len(C_ports)):
        new_pair = (C_ports[i], V_ports[i])
        if new_pair in list(pairs.keys()):
            pairs[new_pair] += 1
        else:
            pairs[new_pair] = 1
            
    multiplicity_list = list(pairs.values())
    if max(multiplicity_list) > 3:
        return RandomaGraphs(n0, Delta_c, Delta_v)
    else:
#         print('original pairs:', pairs)
#         swapped_pairs = DSwitch(pairs)
        swapped_pairs = TSwitch(DSwitch(pairs))
    
    multiplicity_list = list(swapped_pairs.values())
    
    if max(multiplicity_list) > 1:
        print('original pairs:', pairs)
        print('swapped pairs:', swapped_pairs)
        print('Failed to generate graph via edge swapping')
        return None
    
#     print('swapped pairs:', swapped_pairs)
    
    ## Add the paired edges to the graph
    for edge_pair in swapped_pairs.keys():
        G.add_edge(edge_pair[0],edge_pair[1])
        
    return G

def GeneRandGraphsLargeGirth(n0: int, Delta_c: int, Delta_v: int, min_girth: int, min_distance:int, num: int, max_iter:int):
    G_list = []
    n = 0
    i = 0
    while (n < num) and (i < max_iter):
        rand_G = RandomaGraphs(n0 = n0, Delta_c=Delta_c, Delta_v=Delta_v)
        if Girth(rand_G) >= min_girth:
            H = TannerGraphToCheckMat(rand_G)
            d=compute_code_distance(H)
            if d >= min_distance:
                G_list.append(rand_G)
                n += 1
        i += 1
    if (i >= max_iter):
        print('Max iter reached')
        
    return G_list



def SwapEdgePair(G, edge1, edge2):
    assert G.has_edge(edge1[0], edge1[1]) and G.has_edge(edge2[0], edge2[1]), f'edge1 or edge2 is not in G'
    edge1 = (edge1[1], edge1[0]) if edge1[0] < 0 else edge1
    edge2 = (edge2[1], edge2[0]) if edge2[0] < 0 else edge2
    assert G.has_edge(edge1[0], edge1[1]) and G.has_edge(edge2[0], edge2[1]), f'swapped: edge1 or edge2 is not in G'
    G.remove_edge(edge1[0], edge1[1])
    G.remove_edge(edge2[0], edge2[1])
    new_edge1, new_edge2 = (edge1[0], edge2[1]), (edge2[0], edge1[1])
    G.add_edge(new_edge1[0], new_edge1[1])
    G.add_edge(new_edge2[0], new_edge2[1])
    return G

# Randomly swap the edges to increase the girth
def RandSwapEdges1(G, max_iter, target_girth):
    if_success = False
    girth = Girth(G)
    cycles = nx.cycle_basis(G)
    min_cycles = [cycle for cycle in cycles if len(cycle) == girth]
    num_min_cycles = len(min_cycles)

    for j in range(max_iter):
        rand_min_cycle = random.choice(min_cycles)
        #     print('rand min cycle:', rand_min_cycle)
        rand_cycle_index = random.randint(0, len(rand_min_cycle) - 1)
        rand_edge_min_cycle = (rand_min_cycle[rand_cycle_index], rand_min_cycle[(rand_cycle_index + 1)%len(rand_min_cycle)])
        #     print('rand edge min cycle:', rand_edge_min_cycle)
        rand_edge2 = random.choice(list(G.edges()))
        #     print('rand edge 2:', rand_edge2)
        rand_edge_min_cycle = (rand_edge_min_cycle[1], rand_edge_min_cycle[0]) if rand_edge_min_cycle[0] < 0 else rand_edge_min_cycle
        rand_edge2 = (rand_edge2[1], rand_edge2[0]) if rand_edge2[0] < 0 else rand_edge2
        while (rand_edge_min_cycle == rand_edge2):
            rand_edge2 = random.choice(list(G.edges()))
            rand_edge2 = (rand_edge2[1], rand_edge2[0]) if rand_edge2[0] < 0 else rand_edge2

            # Swap the edge pair
        new_G = copy.deepcopy(G)
        new_G = SwapEdgePair(new_G, rand_edge_min_cycle, rand_edge2)
        new_girth = Girth(new_G)
        new_cycles = nx.cycle_basis(new_G)
        new_min_cycles = [cycle for cycle in new_cycles if len(cycle) == new_girth]
        new_num_min_cycles = len(new_min_cycles)
        if (new_girth >= girth) and (new_num_min_cycles <= num_min_cycles):
            G = new_G
            girth = new_girth
            cycles = new_cycles
            min_cycles = new_min_cycles
            num_min_cycles = new_num_min_cycles
#             print('girth:', girth, 'num of min cycles:', num_min_cycles)
        if (girth >= target_girth):
            if_success = True
#             print('succeed')
            return G, if_success
        
        if j == max_iter - 1:
#             print('fail')
            return G, if_success



def GeneRandGraphsLargeGirthFinal(n0: int, Delta_c: int, Delta_v: int, min_girth1: int, target_girth:int, num: int, max_iter:int):
    G_list = []
    n = 0
    i = 0
    while (n < num) and (i < max_iter):
        rand_G = RandomaGraphs(n0 = n0, Delta_c=Delta_c, Delta_v=Delta_v)
        if Girth(rand_G) >= min_girth1:
            max_iter2 = 20000
            swapped_G, if_success = RandSwapEdges1(rand_G, max_iter2, target_girth)
            if if_success:
                G_list.append(swapped_G)
                n += 1
        i += 1
    if (i >= max_iter):
        print('Max iter reached')
        
    return G_list
    