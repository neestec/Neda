# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 00:51:38 2021

@author: Neda
"""

import itertools
import math
import pickle
import random
from copy import deepcopy
from itertools import combinations
from operator import itemgetter
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from numpy import exp
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


def list_node_init():
    """gets layers count as an int variation layer_n
    for each layer creates a random int number of nodes as a member of list_node
    returns list of layer nodes as list_node"""
    inpt = input("input a number as number of layers: ")
    layer_n = int(inpt)
    temp = itertools.count(1)
    index = [next(temp) for i in range(layer_n)]
    # print('index:', index)
    list_node = []
    for i in index:
        #node = np.random.randint(150, 155)
        node = 200
        print('nodes in layer: ', node)
        list_node.append(node)
    print('list_node:', list_node)
    np.save('list_node_initial', list_node, allow_pickle=True)
    np.save('Layen_Count', layer_n, allow_pickle=True)
    return list_node, layer_n


def list_struc(list_node):
    # gets the number of layers, creates and returns struct
    struc = []
    for x in range(len(list_node)):
        str = [x, list_node[x]]
        struc.append(str)

    # print('struc:', struc)
    print('ggggggggggggg')
    np.save('List_Struct', struc, allow_pickle=True)
    return struc


def random_weighted_graph(n):
    """" create a random graph by n= nodes number, p = probability , lower and upper weight"""
    """n = number of nodes in each layer """
    # Erdos renyi graph by poisson degree distribution
    #p = np.random.uniform(0.02, 0.01)
    # p = 0.0495
    # g = nx.erdos_renyi_graph(n, p)


    # Barabasi albert graph by power law distribution
    # m = np.random.randint(1,2)
    # print('neda addade ine:', 1)
    g = nx.generators.barabasi_albert_graph( n, 1)

    # m = g.number_of_edges()
    # print('number of edge:', m)
    # weights = [np.random.randint(5, 10) for r in range(m)]
    uw_edges = g.edges()
    # print ('edges:', uw_edges)
    adj = np.zeros((n, n), dtype="object", order='c')
    # print('zero_adj', adj)
    for edge in uw_edges:
        # print('edge[0]:', edge[0],' edge[1]', edge[1])
        adj[edge[0]][edge[1]] = 1
        adj[edge[1]][edge[0]] = 1
    print('zero_adj after:', adj)
    print('type of g: ', type(g))
    plot_degree_dist(g)
    return adj  # igraph.Graph(uw_edges)


def plot_degree_dist(G):
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees)
    print ('its just here')
    plt.show()
    return


def create_comb_array(list_node):
    strc = list_struc(list_node)  # create a list of layer number and number of nodes in each layer
    # print (strc)
    comb = combinations(strc, 2)  # create all combinations of struct members
    # print(type(comb))
    # print('comb:', comb)
    comb_array = []
    for p in comb:
        print('p:', p)
        comb_array_temp = []
        # comb_array_temp.append(p)
        B0 = p[0][0]
        B1 = p[0][1]
        B2 = p[1][0]
        B3 = p[1][1]
        comb_array_temp.append(B0)
        comb_array_temp.append(B1)
        comb_array_temp.append(B2)
        comb_array_temp.append(B3)
        R = random.randint(1, ((B1 * B3 // 2)))  # random number of edges
        comb_array_temp.append(R)
        comb_array_temp.append(B1 + B3)
        comb_array.append(comb_array_temp)
    # print('comb_array:', comb_array)
    print('comb_dis', comb_array)
    np.save('comb_dis', comb_array, allow_pickle=True)
    return comb_array  # for each member of combarray: 0.comb, 1.B0, 2.B1, 3.B2, 4.B3, 5.R(edge_number), 6.n(lenth)


def create_matrix(list_node):
    """ from list_node() gets list_node as each layer nodes number
    create a zero matrix
    for each layer create a random weighted graph by using of node number
    of layer: random_weighted_graph(nlist[l-1])
    create adjacency matrix for each matrix using g.get_adjacency() """

    n = len(list_node)
    """number of layers , list_node= list of nodes in each layer"""
    total_mtrx = np.zeros((n, n), dtype="object", order='c')
    i = 0
    j = 0
    edge_list = []
    """create diagonal of main matrix """
    for l in range(n):  # n = number of layers
        m = list_node[l]
        adjc_mtrx = random_weighted_graph(m)  # craete random weighted graph for each layer. l = number of nodes
        total_mtrx[i][j] = adjc_mtrx
        i = i + 1
        j = j + 1
    """create Bipartite matrixes"""
    comb = create_comb_array(list_node)
    """for each member of combarray: 0.comb, 1.B0, 2.B1, 3.B2, 4.B3, 5.R(edge_number), 6.n(lenth)"""
    for p in comb:
        print('p in comb: ', p)
        G = nx.bipartite.gnmk_random_graph(p[1], p[3], p[4])
        G_lenth = (p[5])
        Row_order = range(G_lenth)
        G_adj = nx.bipartite.biadjacency_matrix(G, row_order=Row_order, column_order=Row_order)
        G_array = csr_matrix.toarray(G_adj)
        dens_matrx = np.zeros((G_lenth, G_lenth), order='c')
        for i in range(G_lenth):
            for j in range(G_lenth):
                dens_matrx[i][j] = int(G_array[i][j])
        for i in range(0, G_lenth):
            for j in range(0, G_lenth):
                if dens_matrx[i][j] == 1:
                    dens_matrx[j][i] = 1
        total_mtrx[p[0]][p[2]] = dens_matrx
        total_mtrx[p[2]][p[0]] = dens_matrx
    print('total_mtrx', total_mtrx)
    return total_mtrx


def Create_List_of_Nodes(List_Struct):
    list_of_Node = []
    list_of_Node_lables = []
    for n in List_Struct:
        for m in range(n[1]):
            peer = []
            peer.append(n[0])
            peer.append(m)
            list_of_Node.append(peer)
    # print('list of nodes:', list_of_Node)
    for i in range(len(list_of_Node)):
        list_of_Node_lables.append(i)

    np.save('list_of_nodes', list_of_Node, allow_pickle=True)
    np.save('Label', list_of_Node_lables, allow_pickle=True)
    print('list_of_nodes:', list_of_Node)
    return list_of_Node, list_of_Node_lables


def random_atthck_nodes(list_of_nodes):
    # print('len(list_of_nodes):',len(list_of_nodes))

    attacked_number = math.floor(len(list_of_nodes) / 10)
    # print('attacken number on nodes:', attacked_number)
    attacked_list = random.sample(list_of_nodes, attacked_number)
    if len(attacked_list) < 1:
        return list_of_nodes
    # print('attacked nodes:', attacked_list)
    # np.save('Attack_Nodes' , attacked_list , allow_pickle=True)
    return attacked_list


def random_atthck_nodes_GA(list_of_nodes, size):
    # print('len(list_of_nodes):',len(list_of_nodes))
    if len(list_of_nodes) > size:
        attacked_list = random.sample(list_of_nodes, size)
    else:
        attacked_list = list_of_nodes
    return attacked_list


def attacked_node_struct(attacked_nodes):
    # print('---attacked_nodes which passed to attacked_node_struct:', attacked_nodes)
    ls = np.load('List_Struct.npy')
    list_struct = deepcopy(ls)
    node_struct = []
    for node in attacked_nodes:
        node_struct_temp = []
        layer = node[0]
        node_num = node[1]  # index of attacked node in its layer
        Struct = list_struct[layer]  # get struct if the layer that belongs to attacked node
        layer_Node_Number = Struct[1]  # getting Node_Number and wide of adjacency matrix
        node_struct_temp.append(layer)
        node_struct_temp.append(node_num)
        node_struct_temp.append(layer_Node_Number)
        node_struct.append(node_struct_temp)
    return node_struct  # for each node in node_struct as a p: p[0]=layer, p[1]=node_num, p[2]=layer_lenth


def delete_redundant(step_dic, new_list, step):
    is_in = False
    list_recall = []
    step_temp = []
    for s in range(step):
        step_temp = step_dic[s]

        for n in new_list:
            if n in step_temp:
                index = step_temp.index(n)
                new_list.pop(index)
        list_recall = new_list

    return list_recall


def node_Mapping(list_of_Node):
    i = 0
    map_dic = {}
    for node in list_of_Node:
        map_dic[i] = node
        i = i + 1
    print('data type of map_dic ', type(map_dic))
    # create a binary pickle file
    with open('Map_dic.pkl', 'wb') as f:
        pickle.dump(map_dic, f, protocol=pickle.HIGHEST_PROTOCOL)
    # np.save('Map_dic' , map_dic , allow_pickle=False)
    np.save('Total_Node', i, allow_pickle=True)
    print('Map_dic:', map_dic)
    print('Total_node:', i)
    return map_dic, i


def create_index_list(i, total_node):
    with open('Map_dic.pkl', 'rb') as handle:
        map_dic = pickle.load(handle)
        print('map_dic', type(map_dic))
    # map_dic = pickle.load('Map_dic.npy', allow_pickle=True)
    iner_map_dic = deepcopy(map_dic)
    print('data type after convert ', type(iner_map_dic))
    print('iner_map_dic', iner_map_dic)
    print(iner_map_dic[0])
    print(len(iner_map_dic))
    index_list = []
    for a in range(total_node):
        temp_node = iner_map_dic[a]
        if temp_node[0] == i:
            index_list.append(a)
    print('index_list', index_list)
    return index_list


def create_major_matrix(Total_Matrix, Layer_Count):
    # main graph ro misaze
    # Map_dic, Total_Node = node_Mapping(list_of_nodes) / Inha ro darim
    # list_node_initial , Layen_Count = list_node()
    total_node = np.load('Total_Node.npy', allow_pickle=True)
    iner_total_node = deepcopy(total_node)
    main_matrix = np.zeros((iner_total_node, iner_total_node), dtype="int", order='c')
    index_list = []
    temp_node = []
    z = 0
    for i in range(Layer_Count):
        for j in range(Layer_Count):
            matrix = Total_Matrix[i][j]
            if i == j:
                index_list = create_index_list(i, iner_total_node)
                for b in range(len(index_list)):
                    for c in range(len(index_list)):
                        main_matrix[index_list[b]][index_list[c]] = matrix[b][c]


            else:
                bi_index_list = []
                index_list1 = create_index_list(i, iner_total_node)
                index_list2 = create_index_list(j, iner_total_node)
                for mp in index_list2:
                    index_list1.append(mp)
                    bi_index_list = index_list1

                for bi in range(len(bi_index_list)):
                    for ci in range(len(bi_index_list)):
                        if matrix[bi][ci] == 1:
                            main_matrix[bi_index_list[bi]][bi_index_list[ci]] = matrix[bi][ci]

    main = np.matrix(np.array(main_matrix))
    np.save('Main_Matrix', main_matrix, allow_pickle=True)
    return main_matrix


def create_main_graph_init(adjacency_matrix):
    # main graph ro namayesh midim
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    #nx.draw(gr, pos= None, edge_kabel = True)
    #colore_map = []
    # for node in gr:
    #     if node> 10:
    #         colore_map.append('green')
    #     else:
    #         colore_map.append('orange')
    colore_map = ('green')
    nx.draw(gr, node_color = colore_map, node_size=50, with_labels=False)
    #nx.draw(gr, pos= None, ax= None)
    plt.show()
    np.save('Main_Graph', gr, allow_pickle=True)
    return gr


def create_main_graph(adjacency_matrix):
    # main graph ro namayesh midim
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=50, with_labels=False)
    #plt.show()
    return gr


def create_main_graph_copy(adjacency_matrix):
    # main graph ro namayesh midim
    # for i in range(Total_Node):
    #     for j in range(Total_Node):
    #         if adjacency_matrix[i][j] == 1:
    #             adjacency_matrix[j][i] = 0

    rows, cols = np.where(adjacency_matrix == 1)
    edge = (rows.tolist(), cols.tolist())
    edges = zip(rows.tolist(), cols.tolist())
    # print ('yyyyyyyaaaaalllll',edge)

    gr = nx.Graph()
    gr.add_edges_from(edges)
    # nx.draw(gr, node_size=500,  with_labels=True)
    # plt.show()
    np.save('Main_Graph', gr, allow_pickle=True)
    return gr


def closeness_btw(main_graph):
    # closeness ha ro mohasebe mikone
    btw = nx.betweenness_centrality(main_graph, normalized=False)
    return btw


def closeness_deg(main_graph):
    deg_temp = nx.degree(main_graph)
    deg = {}
    for pair in deg_temp:
        deg[pair[0]] = pair[1]
    return deg


def attack_Node_Mapping(attack_list):
    # node haye attack ro be shomare haye jadid map mikone
    total_node = np.load('Total_Node.npy', allow_pickle=True)
    iner_total_node = deepcopy(total_node)
    map_dic = np.load('Map_dic.npy', allow_pickle=True)
    iner_map_dic = deepcopy(map_dic)
    index_list = []
    for a in range(iner_total_node):
        temp_node = iner_map_dic[a]
        for node in attack_list:
            if temp_node == node:
                index_list.append(a)
    return index_list


def connectivity_count_init():
    main_matrix = np.load('Main_Matrix.npy')
    main_graph = create_main_graph_init(main_matrix)
    connectivity = nx.average_node_connectivity(main_graph)
    np.save('Main_Conct', connectivity, allow_pickle=True)
    return connectivity


def connectivity_count(main_graph):
    iner_connectivity = nx.average_node_connectivity(main_graph)
    main_conct = np.load('Main_Conct.npy', allow_pickle=True)
    connectivity = iner_connectivity / main_conct
    return connectivity


def disintegration(node, main_matrix, attack_list):
    # node: the first index of attack node list
    # main_matrix: matrix should updated by each disintegration step
    # attack_List: should update by each disintegration step
    # type = 1: Random / type = 2: DEG / Type = 3: BWN / Type = 4: WGHT
    total_node = np.load('Total_Node.npy', allow_pickle=True)
    iner_total_node = deepcopy(total_node)
    neigh = []
    print('node: ', node)
    for i in range(iner_total_node):
        if main_matrix[i][node] == 1:
            neigh.append(i)
            # print('main_matrix[node][i]: ' , main_matrix[node][i])
            # print('main_matrix[i][node]: ', main_matrix[i][node])
            main_matrix[node][i] = 0
            main_matrix[i][node] = 0
            # print('main_matrix[node][i]: ' , main_matrix[node][i])
            # print('main_matrix[i][node]: ', main_matrix[i][node])

    for n in neigh:
        if n not in attack_list:
            attack_list.append(n)
    final_attack_list = list(set(attack_list))
    print('diiiiiiiiiiiiiiissssssssssssssssssssss')
    return final_attack_list, main_matrix


def closeness_dis_1(type):
    # aval bayad ye peygham neshoon bedim ke in che noe disi hast
    cost_lst = [0.0, 0.0, 0.0, 0.0]
    p = [0.5, 1, 1.5, 2]
    main_matrix = np.load('Main_Matrix.npy', allow_pickle=True)
    init_tota_node = np.load('Total_Node.npy', allow_pickle=True)
    iner_matrix = deepcopy(main_matrix)
    main_graph = create_main_graph(iner_matrix)
    connectivity_lst = []
    connectivity_lst.append(1)
    c_lst = []
    c_lst.append(1)
    main_c = connectivity_count(main_graph)
    attack_list = []
    switcher = {
        1: closeness_btw(main_graph),
        2: closeness_deg(main_graph),
    }
    closeness = switcher.get(type, "Invalid type")
    while len(closeness) != 0:
        switcher = {
            1: closeness_btw(main_graph),
            2: closeness_deg(main_graph),
        }
        closeness = switcher.get(type, "Invalid type")
        sort_order = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
        if len(closeness) == 0:
            print('final main matrix for other methodes: ', iner_matrix)
            print('Network has disintegrated successfuly')
            np.save('conct_btw_lst.npy', connectivity_lst)
            np.save('cost_btw.npy', cost_lst)
            np.save('cc_btw.npy', c_lst)
            return connectivity_lst, cost_lst
        else:
            if len(closeness) != 0 and len(attack_list) == 0:
                max_order_node = sort_order[0][0]
            for node in attack_list:
                if node not in closeness:
                    index = attack_list.index(node)
                    attack_list.pop(index)
                    print('alone node hase deleted: ', node)

            max_order_node = sort_order[0][0]
            print('target node: ', max_order_node)
            for i in range(len(p)):
                cost = cost_count(main_graph, [max_order_node], p[i])
                cost_lst[i] = cost_lst[i] + cost[0][1]

            attack_list, iner_matrix = disintegration(max_order_node, iner_matrix, attack_list)
            print('iner_matrix in closeness recursive dis:', "\n", iner_matrix)
            main_graph = create_main_graph(iner_matrix)
            active_nodes = active_node(iner_matrix)
            connectivity = len(active_nodes) / init_tota_node
            c = connectivity_count(main_graph)
            c_lst.append(c / main_c)
            connectivity_lst.append(connectivity)

            print('connectivity_lst', connectivity_lst)


def degree_average ():
    main_matrix = np.load('Main_Matrix.npy', allow_pickle=True)
    init_tota_node = np.load('Total_Node.npy', allow_pickle=True)
    iner_matrix = deepcopy(main_matrix)
    main_graph = create_main_graph(iner_matrix)
    degree_closeness = closeness_deg(main_graph)
    sort_order = sorted(degree_closeness.items(), key=lambda x: x[1], reverse=True)
    print('degree_closeness: ', degree_closeness)
    print ('sort_rder : ', sort_order)
    print ('index: ', sort_order[0][1])
    print (len (sort_order))
    sum= 0
    for i in sort_order:
        sum = sum + i[1]
    avrg = sum / len(sort_order)
    print ('avrage = ', avrg)
    return degree_closeness

def closeness_dis_2(type):
    # aval bayad ye peygham neshoon bedim ke in che noe disi hast
    cost_lst = [0.0, 0.0, 0.0, 0.0]
    p = [0.5, 1, 1.5, 2]
    main_matrix = np.load('Main_Matrix.npy', allow_pickle=True)
    init_tota_node = np.load('Total_Node.npy', allow_pickle=True)
    iner_matrix = deepcopy(main_matrix)
    main_graph = create_main_graph(iner_matrix)
    connectivity_lst = []
    connectivity_lst.append(1)
    c_lst = []
    c_lst.append(1)
    attack_nodes_serie = []
    main_c = connectivity_count(main_graph)
    attack_list = []
    switcher = {
        1: closeness_btw(main_graph),
        2: closeness_deg(main_graph),
    }
    closeness = switcher.get(type, "Invalid type")
    while len(closeness) != 0:
        switcher = {
            1: closeness_btw(main_graph),
            2: closeness_deg(main_graph),
        }
        closeness = switcher.get(type, "Invalid type")
        sort_order = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
        print ('@@@@@@@@@ : ' , sort_order)
        if len(closeness) == 0:
            print('final main matrix for other methodes: ', iner_matrix)
            print('Network has disintegrated successfuly')
            print('attack_node_serie: ', attack_nodes_serie)
            np.save('conct_deg_lst.npy', connectivity_lst)
            np.save('attack_node_series', attack_nodes_serie)
            print('attack_len: ', len(attack_nodes_serie), 'conectivity_len: ', len(connectivity_lst))
            print('cost_list: ', cost_lst)
            np.save('cost_deg.npy', cost_lst)
            np.save('cc_deg.npy', c_lst)
            return connectivity_lst, cost_lst
        else:
            if len(closeness) != 0 and len(attack_list) == 0:
                max_order_node = sort_order[0][0]
            for node in attack_list:
                if node not in closeness:
                    index = attack_list.index(node)
                    attack_list.pop(index)
                    print('alone node hase deleted: ', node)

            max_order_node = sort_order[0][0]
            attack_nodes_serie.append(max_order_node)
            print('target node: ', max_order_node)
            for i in range(len(p)):
                cost = cost_count(main_graph, [max_order_node], p[i])
                cost_lst[i] = cost_lst[i] + cost[0][1]

            attack_list, iner_matrix = disintegration(max_order_node, iner_matrix, attack_list)
            print('iner_matrix in closeness recursive dis:', "\n", iner_matrix)
            main_graph = create_main_graph(iner_matrix)
            active_nodes = active_node(iner_matrix)
            connectivity = len(active_nodes)
            conct = (connectivity / init_tota_node)
            c = connectivity_count(main_graph)
            c_lst.append(c / main_c)
            connectivity_lst.append(conct)

            print('connectivity_lst', connectivity_lst)


def random_recursive_dis():
    cost_lst = [0.0, 0.0, 0.0, 0.0]
    p = [0.5, 1.0, 1.5, 2]
    main_matrix = np.load('Main_Matrix.npy', allow_pickle=True)
    iner_matrix = deepcopy(main_matrix)
    init_tota_node = np.load('Total_Node.npy', allow_pickle=True)
    main_graph = create_main_graph(iner_matrix)
    connectivity_lst = []
    connectivity_lst.append(1)
    c_lst = []
    c_lst.append(1)
    main_c = connectivity_count(main_graph)
    attack_list = []
    closeness = closeness_deg(main_graph)
    while len(closeness) != 0:
        closeness = closeness_deg(main_graph)
        print('closeness before sorting: ', closeness)
        sort_order = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
        print('sorted:::', sort_order)
        if len(closeness) == 0:
            print('final main matrix for other methodes: ', iner_matrix)
            print('Network has disintegrated successfuly')
            np.save('conct_rand_lst.npy', connectivity_lst)
            np.save('cost_rand.npy', cost_lst)
            np.save('cc_rand.npy', c_lst)
            return connectivity_lst, cost_lst
        else:
            if len(closeness) != 0 and len(attack_list) == 0:
                rand_order_node = sort_order[np.random.randint(0, len(sort_order))][0]
            for node in attack_list:
                if node not in closeness:
                    index = attack_list.index(node)
                    attack_list.pop(index)
                    print('alone node hase deleted: ', node)
            # sort_order , max_order, attack_list_rand = attack_Node_Ordering(attack_list, closeness )

            rand_order_node = sort_order[np.random.randint(0, len(sort_order))][0]

            print('target node: ', rand_order_node)
            for i in range(len(p)):
                cost = cost_count(main_graph, [rand_order_node], p[i])
                cost_lst[i] = cost_lst[i] + cost[0][1]
            attack_list, iner_matrix = disintegration(rand_order_node, iner_matrix, attack_list)
            main_graph = create_main_graph(iner_matrix)
            active_nodes = active_node(iner_matrix)
            connectivity = len(active_nodes)
            conct = (connectivity / init_tota_node)
            c = connectivity_count(main_graph)
            c_lst.append(c / main_c)
            connectivity_lst.append(conct)


def weight_def(main_matrix):
    # be ezaye har node ye vazne tasadofi ijad mikone va liste node haye faal ro ham tashkhis mide va barmigardoone
    list_of_weight = []
    total_node = np.load('Total_Node.npy', allow_pickle=True)
    iner_total_node = deepcopy(total_node)
    for i in range(iner_total_node):
        for j in range(iner_total_node):
            node1 = []
            if main_matrix[i][j] == 1:
                weight = np.random.randint(5, 20)
                node1.append(i)
                node1.append(j)
                node1.append(weight)
                list_of_weight.append(node1)
    # print('list of weight :: ' , list_of_weight)
    # baraye yalha be soorate do tarafe vazn injad shode bood ke too in halgheha yektarafash kardam.
    for node in list_of_weight:
        i = node[0]
        j = node[1]
        for n in list_of_weight:
            if n[0] == j and n[1] == i:
                n[2] = node[2]
    print('list_of_weight: ', list_of_weight)
    np.save('Triple_Weight', list_of_weight, allow_pickle=True)
    return list_of_weight


def active_node_init(main_matrix):
    # har bar ke matrix ro update mikonm va azash kam mishe in metode mire node haye zendash ro list mikone
    total_node = np.load('Total_Node.npy', allow_pickle=True)
    iner_total_node = deepcopy(total_node)
    active_node = []
    for i in range(iner_total_node):
        for j in range(iner_total_node):
            node1 = []
            if main_matrix[i][j] == 1 and main_matrix[j][i] == 1:
                node1.append(i)
                node1.append(j)
                active_node.append(node1)
    active_node_list = []
    for node in active_node:
        if node[0] not in active_node_list:
            active_node_list.append(node[0])
    np.save('Active_Node', active_node_list, allow_pickle=True)
    return active_node_list


def active_node(main_matrix):
    # har bar ke matrix ro update mikonm va azash kam mishe in metode mire node haye zendash ro list mikone
    total_node = np.load('Total_Node.npy', allow_pickle=True)
    iner_total_node = deepcopy(total_node)
    active_node = []
    for i in range(iner_total_node):
        for j in range(iner_total_node):
            node1 = []
            if main_matrix[i][j] == 1 and main_matrix[j][i] == 1:
                node1.append(i)
                node1.append(j)
                active_node.append(node1)
    active_node_list = []
    for node in active_node:
        if node[0] not in active_node_list:
            active_node_list.append(node[0])
    # np.save('Active_Node' , active_node_list ,  allow_pickle=True)
    return active_node_list


def average_count(list_node):
    # ba estefade az vazne yalhayi ke be node vaslan miyangin migire
    # print ('list_node_out: ', list_node)
    # print('list_node in average_count :', list_node)
    count = 0
    sum = 0
    for node in list_node:
        # print ('Node: ', node)
        count = count + 1
        sum = sum + node[2]
    # print('node: ', node, 'count:', count , 'sum:', sum )
    weight_avr = (sum / count)
    # print(sum , '/', count, '=' , weight_avr)

    return weight_avr


def average_count_copy(list_node):
    # ba estefade az vazne yalhayi ke be node vaslan miyangin migire
    # print ('list_node_out: ', list_node)
    # print('list_node in average_count :', list_node)
    count = 0
    sum = 0
    for node in list_node:
        # print ('Node: ', node)
        count = count + 1
        sum = sum + node[2]
    # print('node: ', node, 'count:', count , 'sum:', sum )
    weight_avr = (sum / count)
    return weight_avr


def weight_account_init(list_of_weight, active_nodes):
    # vazn node haye active ro hesab mikone(methodesh ro farakhani mikone)
    # va tahesh ham list ro sort mikone
    node_and_avr_list = []
    for i in active_nodes:
        list_node_internal = []
        node_and_avr_temp = []
        for node in list_of_weight:
            if i == node[0]:
                list_node_internal.append(node)
        node_avr = average_count(list_node_internal)
        node_and_avr_temp.append(node_avr)
        node_and_avr_temp.append(i)
        node_and_avr_list.append(node_and_avr_temp)
    node_and_avr_list.sort()
    node_and_avr_list.reverse()
    np.save('Averg_Weight', node_and_avr_list, allow_pickle=True)
    return node_and_avr_list


def table_initiator_Q(total_node):
    # Create Q-Table_Total by Total_node Dimentions
    # n = len(total_node)
    q_table_total = np.zeros((total_node, total_node), dtype="float", order='c')
    sum_value = []
    np.save('Q_table.npy', q_table_total)
    np.save('Sum_valu.npy', sum_value)

    return q_table_total


def h_table_initiator(total_node):
    # Create H-Table by Total_node Dimentions
    # n = len(total_node)
    h_table = np.zeros((total_node, total_node), dtype="float", order='c')
    sum_valuen = []
    last_browse = [0]
    print('h_table', h_table)
    np.save('H_Table.npy', h_table)
    np.save('Last_automata_brows.npy', last_browse)
    return h_table


def weight_account(list_of_weight, active_nodes):
    # vazn node haye active ro hesab mikone(methodesh ro farakhani mikone)
    # va tahesh ham list ro sort mikone
    node_and_avr_list = []
    for i in active_nodes:
        list_node_internal = []
        node_and_avr_temp = []
        for node in list_of_weight:
            if i == node[0]:
                list_node_internal.append(node)
        node_avr = average_count(list_node_internal)
        node_and_avr_temp.append(node_avr)
        node_and_avr_temp.append(i)
        node_and_avr_list.append(node_and_avr_temp)
    node_and_avr_list.sort()
    node_and_avr_list.reverse()
    return node_and_avr_list


def weight_account_copy(list_of_weight, active_nodes):
    # be ezaye list haeye vazn ha average vazn yek node ro hesab mikone va tahesh ham list ro sort mikone
    # print('list_of_weight:::::::::::::' , list_of_weight)
    # print(active_nodes)
    node_and_avr_list = []

    for i in active_nodes:
        list_node_internal = []
        node_and_avr_temp = []
        for node in list_of_weight:
            if i == node[0]:
                list_node_internal.append(node)
        # print('list_node_internal:::::::::::::',list_node_internal)
        node_avr = average_count_copy(list_node_internal)
        node_and_avr_temp.append(node_avr)
        node_and_avr_temp.append(i)
        node_and_avr_list.append(node_and_avr_temp)

    node_and_avr_list.sort()
    # = sorted(node_and_avr_list)
    node_and_avr_list.reverse()
    # print('node_and_avr_list::::::::',node_and_avr_list)
    return node_and_avr_list


def attack_weight_sort(attack_node, node_avrg):
    # list attack ro bar asase vazn sort mikone va khoroojish faghat yek node hast
    # print('attack_nodes in attack weight sort: ', attack_node)
    # print ('node_avrg in attack weight sort: ', node_avrg)
    attack_sort = []
    for node in attack_node:
        internal_node = []
        for nd in node_avrg:
            # print('-------', node , '==============',nd )
            if node == nd[1]:
                internal_node.append(nd[0])
                internal_node.append(node)
                attack_sort.append(internal_node)
                # print('gdfhgsdfhgdfheg' , attack_sort)

    attack_sort.sort()
    attack_sort.reverse()
    # print('attack_sort: ',attack_sort)
    final_sorted_attack_node = []
    for node in attack_sort:
        final_sorted_attack_node.append(node[1])
    # print('final_sorted_attack_node: ', final_sorted_attack_node)
    return final_sorted_attack_node


def attack_weight_sort_copy(attack_node, node_avrg):
    # list attack ro bar asase vazn sort mikone va khoroojish faghat yek node hast
    attack_sort = []
    for node in attack_node:
        internal_node = []
        for nd in node_avrg:
            if node == nd[1]:
                internal_node.append(nd[0])
                internal_node.append(node)
                attack_sort.append(internal_node)

    attack_sort.sort()
    attack_sort.reverse()
    # print('attack_sort: ',attack_sort)
    final_sorted_attack_node = []
    for node in attack_sort:
        final_sorted_attack_node.append(node[1])
    # print('final_sorted_attack_node: ', final_sorted_attack_node)
    return final_sorted_attack_node


def weight_recursive_dis():
    # recursive disintegration ro anjam mide
    # iner_main_matrix = [row[:] for row in main_matrix]
    cost_lst = [0.0, 0.0, 0.0, 0.0]
    p = [0.5, 1.0, 1.5, 2]
    main_matrix = np.load('Main_Matrix.npy', allow_pickle=True)
    iner_main_matrix = deepcopy(main_matrix)
    init_tota_node = np.load('Total_Node.npy', allow_pickle=True)
    main_graph = create_main_graph(iner_main_matrix)
    connectivity_lst = []
    connectivity_lst.append(1)
    c_lst = []
    c_lst.append(1)
    main_c = connectivity_count(main_graph)
    attack_list = []
    avrag_weight = np.load('Averg_Weight.npy', allow_pickle=True)
    iner_averg_weigh = deepcopy(avrag_weight)
    print('iner_averg_weigh', iner_averg_weigh)

    active_nodes = active_node(iner_main_matrix)
    while len(active_nodes) != 0:
        for node in attack_list:
            if node not in active_nodes:
                index = attack_list.index(node)
                attack_list.pop(index)
                print('alone node hase deleted: ', node)
        if len(active_nodes) != 0 and len(attack_list) == 0:

            # baraye jologiri az khata ya loop binahayat da moredi ke graph az aval chand bakhshi boode
            main_graph = create_main_graph(iner_main_matrix)
            triple_weight = np.load('Triple_Weight.npy', allow_pickle=True)
            list_of_weight = deepcopy(triple_weight)
            primitive_list_of_weight = deepcopy(list_of_weight)
            node_averg = weight_account(list_of_weight, active_nodes)
            primitive_node_avrg = deepcopy(node_averg)
            attack_list.append(node_averg[0][1])
        else:
            if len(attack_list) != 0:
                print('44444444444444444')
                target_node = attack_list[0]
                print('5555555555')
                for i in range(len(p)):
                    cost = cost_count(main_graph, [target_node], p[i])
                    cost_lst[i] = cost_lst[i] + cost[0][1]
                attack_list, iner_main_matrix = disintegration(target_node, iner_main_matrix, attack_list)
                main_graph = create_main_graph(iner_main_matrix)
                active_nodes = active_node(iner_main_matrix)
                connectivity = len(active_nodes)
                conct = (connectivity / init_tota_node)
                connectivity_lst.append(conct)
                c = connectivity_count(main_graph)
                c_lst.append(c / main_c)
                # print('attack_list', attack_list)
                # list_of_weight  = weight_def (iner_main_matrix)
                active_nodes = active_node(iner_main_matrix)
                node_averg = weight_account(list_of_weight, active_nodes)
                attack_list = attack_weight_sort(attack_list, node_averg)
    if len(active_nodes) == 0:
        print('Network has disintegrated successfuly by wight method ')
        np.save('conct_weight_lst.npy', connectivity_lst)
        np.save('cost_weight.npy', cost_lst)
        np.save('cc_weight.npy', c_lst)
        return connectivity_lst, cost_lst


def attack_maping(attack_list, map_dic):
    attack_map = []
    for node in attack_list:
        for n in map_dic:
            if node == map_dic[n]:
                attack_map.append(n)
    np.save('Attack_Map', attack_map, allow_pickle=True)
    print('attack_map::::::::::', attack_map)
    return attack_map


def normalize(abnormal_list):
    ab_normal_list = deepcopy(abnormal_list)
    # list ha jofti hastan. aval bayad yek ozvishoon kinim bad aza ro normal konim, bad dobare set konim.
    # nokteye mohem ine li nabayad az index 1 aza be onvan index join shodan estefade konim chon ina shomare node ha
    # hastan va daem dar hale taghir.
    # print ('abnormal list',ab_normal_list)
    if len(ab_normal_list) != 0:
        node = []
        point = []
        for n in ab_normal_list:
            node.append(n[0])
            point.append(n[1])
        # print ('point' , point)
        # print ('Node: ', node)
        norm_list = []
        min_value = min(point)
        max_value = max(point)
        for value in point:
            if max_value != min_value:
                tmp = (value - min_value) / (max_value - min_value)
                norm_list.append(tmp)
            else:
                tmp = 0
                norm_list.append(tmp)
        # print('Normalized List:',norm_list)
        normal_list_final = []
        for i in range(len(ab_normal_list)):
            internal_point = []
            internal_point.append(node[i])
            internal_point.append(norm_list[i])
            normal_list_final.append(internal_point)
        # print('normal_list_final', normal_list_final)
        return normal_list_final
    else:
        return []


# ------------------ GA-------------

def fitness_count(Averg_Weight, main_graph, initiator):
    """just list normalize"""
    weight_list_avrg = deepcopy(Averg_Weight)

    bc = closeness_btw(main_graph)
    dc = closeness_deg(main_graph)
    bc_sort = sorted(bc.items(), key=lambda x: x[1], reverse=True)
    dc_sort = sorted(dc.items(), key=lambda x: x[1], reverse=True)
    # weight_list_avrg = weight_account_copy(weight_list_triple , attack_list)
    print('weight: ', weight_list_avrg, "\n", 'bc: ', bc_sort, "\n", 'dc', dc_sort)

    weight_list_reverse = []
    for n in weight_list_avrg:
        temp_n = []
        temp_n.append(n[1])
        temp_n.append(n[0])
        weight_list_reverse.append(temp_n)
    print('weight_list_reverse', weight_list_reverse)
    weight_normal = normalize(weight_list_reverse)
    bc_normal = normalize(bc_sort)
    dc_normal = normalize(dc_sort)
    # motabeghe node haye attack az bd , dc , weight mikeshe biron
    weight_lst = []
    bc_lst = []
    dc_lst = []
    # create attack list from bc dc uw
    for n in initiator:
        for node in weight_normal:
            if n == node[0]:
                weight_lst.append(node)
    for n in initiator:
        for node in bc_normal:
            if n == node[0]:
                bc_lst.append(node)
    for n in initiator:
        for node in dc_normal:
            if n == node[0]:
                dc_lst.append(node)
    print('len initiator:', len(initiator))
    print('methode: fitness_count: ', 'len weight_lst:', len(weight_lst), 'len dc_lst:', len(dc_lst), 'len bc_lst: ',
          len(bc_lst))
    print('len in fitness_count:')
    print('len(weight_lst):', len(weight_lst), "\n", 'len(dc_lst)', len(dc_lst), "\n", 'len(bc_lst)', len(bc_lst))
    return weight_lst, dc_lst, bc_lst


def fitness_arrenge(uw, dc, bc, initiator):
    """arrange nodes by fitness value"""
    print('bc:', bc, "\n", 'dc:', dc, "\n", 'uw:', uw)
    bc_int = []
    for num in bc:
        num[0] = int(num[0])
        bc_int.append(num)
    dc_int = []
    for num in dc:
        num[0] = int(num[0])
        dc_int.append(num)
    uw_int = []
    for num in uw:
        num[0] = int(num[0])
        uw_int.append(num)
    create_dataset = False
    for i in range(len(bc_int)):
        # print('i:', i)
        # print('bc_int[i][0] : ', bc_int[i][0])
        # print('dc_int[i][0] : ', dc_int[i][0])
        # print('uw_int[i][0] : ', uw_int[i][0])
        # print('bc len: ', len(bc_int))
        # print('dc len: ', len(dc_int))
        # print('uw len: ', len(uw_int))
        if bc_int[i][0] == dc_int[i][0] and bc_int[i][0] == uw_int[i][0]:
            create_dataset = True
        # else:
        #     print('index ha ba ham yeki nistan', bc_int[i][0], dc_int[i][0],  uw[i][0])
        #     for node in uw:
        #         if node[i][0] == uw[i][0]:
        #             index = uw.index(node)
        #             uw.pop(index)
    if len(bc_int) == len(dc_int) and len(dc_int) == len(uw_int):
        print('toolha ba ham barabaran')
    else:
        print('toolha ba ham yeki nist', len(bc_int), len(dc_int), len(uw_int))
        return

    if create_dataset:
        node_number = []
        bc_point = []
        dc_point = []
        uw_point = []
        sum_point = []
        for node in bc_int:
            node_number.append(node[0])
            bc_point.append(node[1])
        for node in dc_int:
            dc_point.append(node[1])
        for node in uw_int:
            uw_point.append(node[1])
        for i in range(len(bc_int)):
            local_sum = dc_point[i] + bc_point[i] + uw_point[i]
            sum_point.append(local_sum)
        data_frame = pd.DataFrame({
            "node_number": node_number,
            "bc": bc_point,
            "dc": dc_point,
            "uw": uw_point,
            "sum": sum_point
        })
        # data_frame.sort_values("sum")
        data_frame = data_frame.sort_values(by=['sum'], ascending=True)
        print('data_frame after sorting', data_frame)
        column = data_frame["sum"]
        node_lst = []
        node_lst = data_frame['node_number'].tolist()
        max_sum_value = column.max()
        target_node = data_frame['node_number'][data_frame[data_frame['sum'] == max_sum_value].index.tolist()].tolist()
        # indx = data_frame.loc[data_frame['sum'] == max_sum_value, index]
        print('max_sum_value:', max_sum_value, 'target_node', target_node)
        print(type(node_lst))
        print('node_lst', node_lst)
        return node_lst


def list_split(initiate_lst, initial_len, mutation_portion, crossover_portion):
    """split list utilizing portions even with longe lenth. input is a sorted list& lent = initial"""
    print('initiated list in split:', initiate_lst)
    iner_init_cross_lst = deepcopy(initiate_lst)

    # be dast avardane tedade har ghesmat az toole jhen (initial_len)
    mutation_count = math.ceil(mutation_portion * initial_len)
    parent_count = 1  # initial_len - mutation_count-cross_count
    cross_count = (initial_len - (mutation_count - parent_count) - 2)
    print('mutation_count:', mutation_count)
    print('cross_count:', cross_count)
    print('parent_count:', parent_count)

    # be dast avardane tedad har ghesmat az toole reshteye voroodi (initial_lst)
    mutation_of_inpunt_lst = math.ceil(mutation_portion * len(initiate_lst))
    parent_of_input_lst = 1
    cross_of_input_lst = math.ceil(crossover_portion * len(initiate_lst))

    # create lists
    mut_lst = []
    for i in range(mutation_of_inpunt_lst):
        mut_lst.append(initiate_lst[i])
    print('mut_lst:', mut_lst)
    # create cross members after pop mut and befor pop parent
    for node in mut_lst:
        if node in iner_init_cross_lst:
            index = iner_init_cross_lst.index(node)
            iner_init_cross_lst.pop(index)
    print('base after pop mut: ', iner_init_cross_lst)
    mut = random.sample(mut_lst, mutation_count)
    parent = []
    parent.append(iner_init_cross_lst[-1])
    print('parent:', parent)
    for node in parent:
        if node in iner_init_cross_lst:
            index = iner_init_cross_lst.index(node)
            iner_init_cross_lst.pop(index)
    if len(iner_init_cross_lst) > cross_count:
        children = random.sample(iner_init_cross_lst, cross_count)
    else:
        children = iner_init_cross_lst
    print('mut_cross_lst:', mut_lst, 'parent_final:', parent, 'children:', children)
    cross_mut_child = []
    last_gen = []
    for i in mut:
        cross_mut_child.append(i)
    for i in children:
        cross_mut_child.append(i)
    for i in cross_mut_child:
        last_gen.append(i)
    for i in parent:
        last_gen.append(i)
    return mut, parent, children, cross_mut_child, last_gen


def crossover(cross_mut_child, main_martix, last_gen):
    """hamsaye haye azaye list ra barmigardanad
       liste voroodi mut+ child ast"""
    total_node = np.load('Total_Node.npy', allow_pickle=True)
    # for num in children:
    #     num = int(num)
    print('cross_mut_child:', cross_mut_child)
    iner_total_node = deepcopy(total_node)
    neigh = []
    neigh_single = []
    neigh_final = []
    for i in cross_mut_child:
        # print('i in children: ', i)
        for j in range(iner_total_node):
            # print('j in range(iner_total_node) : ', j)
            if main_martix[j][i] == 1:
                neigh.append(j)

    for i in neigh:
        if i not in neigh_single:
            neigh_single.append(i)

    for i in neigh_single:
        if i in last_gen:
            index = neigh_single.index(i)
            neigh_single.pop(index)

    crossover_lst = list(set(neigh_single))
    print('neighhhhhhhhhhhhhhhh:', neigh)
    print('neigh len:', len(neigh))
    print('len neigh_single after revising: ', len(neigh_single))
    print('final list of neigh in crossover: ', neigh_single)
    print('crossoverrrrrrrrrrrr', crossover_lst)
    return crossover_lst


def list_initiate(main_matrix):
    iner_matrix = deepcopy(main_matrix)
    main_graph = create_main_graph(iner_matrix)
    active_nodes = active_node(iner_matrix)
    initiator = random_atthck_nodes(active_nodes)
    generation_size = len(initiator)
    print('methode: list_initiate: ', 'len initiate list: ', len(initiator), "\n", 'initiate list:', initiator)
    return initiator, generation_size


def list_sorting(weight_Triple, main_matrix, main_graph, initiate_lst):
    iner_active_node = []
    sorted_lst = []
    sorted_lst_int = []
    iner_active_node = active_node(main_matrix)
    print('len(iner_active_node): ', len(iner_active_node))
    if len(iner_active_node) < 3:
        sorted_lst = deepcopy(iner_active_node)
        for num in sorted_lst:
            num = int(num)
            sorted_lst_int.append(num)
        print('****************** node ha tamoom shodan ********************')
        return sorted_lst_int
    Averg_Weight = weight_account(weight_Triple, iner_active_node)
    weight_lst, dc_lst, bc_lst = fitness_count(Averg_Weight, main_graph, initiate_lst)
    # bc , dc, uw = list_allignment(bc_lst, dc_lst, weight_lst)
    sorted_lst = fitness_arrenge(weight_lst, dc_lst, bc_lst, initiate_lst)

    for num in sorted_lst:
        num = int(num)
        sorted_lst_int.append(num)
    print('sorted lst in list_sorting: ', sorted_lst)
    return sorted_lst_int


def list_constructor(mut, new_mut, permanent_parent_lst, new_per_par, new_children):
    GA_initiator = []
    for i in mut:
        GA_initiator.append(i)
    for i in new_mut:
        GA_initiator.append(i)
    for i in permanent_parent_lst:
        GA_initiator.append(i)
    for i in new_per_par:
        GA_initiator.append(i)
    for i in new_children:
        GA_initiator.append(i)
    return GA_initiator


def GA_target_node(mutation_portion, crossover_portion, initiate_lst,
                   generation_size, main_graph, main_matrix, evolution, weight_triple):
    iner_init = deepcopy(initiate_lst)
    if len(iner_init) < 3:
        print('list is shorter than generation')
        target_node = iner_init[-1]
        return target_node
    iner_matrix = deepcopy(main_matrix)
    z = 0
    while z < evolution:
        if len(iner_init) < generation_size:
            print('list is shorter than generation')
            target_node = iner_init[-1]
            return target_node
        # gen avaliya ro dorost mikone
        last_gen = list_sorting(weight_triple, iner_matrix, main_graph, iner_init)
        print('last_gen in first step:', last_gen)
        target_node = last_gen[-1]
        print('target_node on first step:', target_node)

        mut, permanent_parent_lst, children, cross_mut_child, last_gen = \
            list_split(last_gen, generation_size, mutation_portion, crossover_portion)
        print('children in target node: ', children)
        children_len = len(children)
        crossover_lst = crossover(cross_mut_child, iner_matrix, last_gen)
        print('crossover before split in target_node methode: ', crossover_lst)
        for i in permanent_parent_lst:
            crossover_lst.append(i)
        crossover_lst_sorted = list_sorting(weight_triple, iner_matrix, main_graph, crossover_lst)
        crossover_lst_len = len(crossover_lst)
        mut, permanent_parent_lst, children, cross_mut_child, last_gen = \
            list_split(crossover_lst_sorted, generation_size, mutation_portion, crossover_portion)
        print('last_gen:', last_gen)
        target_node = last_gen[-1]
        z += 1
        print('z::::', z)
        print('target_node:', target_node)
    return target_node


def GA_dis(cross_portion, mutation_portion, evolution):
    cost_lst = [0.0, 0.0, 0.0, 0.0]
    p = [0.5, 1.0, 1.5, 2]
    # initiator = deepcopy(primitive)
    # init_len = len(initiator)
    main_matrix = np.load('Main_Matrix.npy', allow_pickle=True)
    iner_matrix = deepcopy(main_matrix)
    init_tota_node = np.load('Total_Node.npy', allow_pickle=True)
    Averg_Weight = np.load('Averg_Weight.npy', allow_pickle=True)
    Triple_Weight = np.load('Triple_Weight.npy', allow_pickle=True)
    main_graph = create_main_graph(iner_matrix)
    connectivity_lst = []
    connectivity_lst.append(1)
    c_lst = []
    c_lst.append(1)
    main_c = connectivity_count(main_graph)
    active_nodes = active_node(iner_matrix)
    while len(active_nodes) != 0:

        if len(active_nodes) == 0:
            print('Network has disintegrated successfuly in GA')
            np.save('conct_GA_lst.npy', connectivity_lst)
            np.save('cost_GA.npy', cost_lst)
            np.save('cc_GA.npy', c_lst)
            return connectivity_lst, cost_lst

        initiate_lst, generation_size = list_initiate(iner_matrix)
        if len(initiate_lst) == 0:
            print('Network has disintegrated successfuly in GA')
            np.save('conct_GA_lst.npy', connectivity_lst)
            np.save('cost_GA.npy', cost_lst)
            np.save('cc_GA.npy', c_lst)
            return connectivity_lst, cost_lst

        for node in initiate_lst:
            if node not in active_nodes:
                index = initiate_lst.index(node)
                initiate_lst.pop(index)
                print('alone node hase deleted: ', node)
        if len(initiate_lst) < 6:
            print('list is shorter than generation')
            target_node = initiate_lst[-1]
            print('target_node in GA_dis:', target_node)
            for i in range(len(p)):
                cost = cost_count(main_graph, [target_node], p[i])
                cost_lst[i] = cost_lst[i] + cost[0][1]
            attack_lst, iner_matrix = disintegration(target_node, iner_matrix, [])
            active_nodes = active_node(iner_matrix)
            print('active_node after dis:', active_nodes)
            main_graph = create_main_graph(iner_matrix)
            connectivity = len(active_nodes)
            conct = (connectivity / init_tota_node)
            connectivity_lst.append(conct)
            if len(active_nodes) == 0:
                print('Network has disintegrated successfuly in GA')
                np.save('conct_GA_lst.npy', connectivity_lst)
                np.save('cost_GA.npy', cost_lst)
                np.save('cc_GA.npy', c_lst)
                return connectivity_lst, cost_lst
        else:

            target_node = GA_target_node(mutation_portion, cross_portion, initiate_lst, generation_size, main_graph,
                                         iner_matrix, evolution, Triple_Weight)
            print('target_node in GA_dis:', target_node)
            for i in range(len(p)):
                cost = cost_count(main_graph, [target_node], p[i])
                cost_lst[i] = cost_lst[i] + cost[0][1]
            attack_lst, iner_matrix = disintegration(target_node, iner_matrix, [])
            active_nodes = active_node(iner_matrix)
            print('active_node after dis:', active_nodes)
            main_graph = create_main_graph(iner_matrix)
            connectivity = len(active_nodes)
            conct = (connectivity / init_tota_node)
            connectivity_lst.append(conct)
            c = connectivity_count(main_graph)
            c_lst.append(c / main_c)
            if len(active_nodes) == 0:
                print('Network has disintegrated successfuly in GA')
                np.save('conct_GA_lst.npy', connectivity_lst)
                np.save('cost_GA.npy', cost_lst)
                np.save('cc_GA.npy', c_lst)
                return connectivity_lst, cost_lst
    return connectivity_lst, cost_lst


# ___________________Greedy_____________
def data_preparing(main_graph, weight_list_avrg):
    bc = closeness_btw(main_graph)
    dc = closeness_deg(main_graph)
    bc_sort = sorted(bc.items(), key=lambda x: x[1], reverse=True)
    dc_sort = sorted(dc.items(), key=lambda x: x[1], reverse=True)
    weight_list_reverse = []
    for n in weight_list_avrg:
        temp_n = []
        temp_n.append(n[1])
        temp_n.append(n[0])
        weight_list_reverse.append(temp_n)
    print('weight_list_reverse', weight_list_reverse)
    weight_list_reverse_sort = sorted(weight_list_reverse, key=itemgetter(0))
    print('weight_list_reverse_sort:', weight_list_reverse_sort)
    weight_normal = normalize(weight_list_reverse_sort)
    bc_normal = normalize(bc_sort)
    dc_normal = normalize(dc_sort)
    bc_normal = sorted(bc_normal, key=itemgetter(0))
    dc_normal = sorted(dc_normal, key=itemgetter(0))
    print('bc_normal: ', bc_normal, "\n", 'dc_normal: ', dc_normal, "\n", 'weight_normal: ', weight_normal)
    print(len(bc_normal), len(dc_normal), len(weight_normal))
    return bc_normal, dc_normal, weight_normal


def target_choose(bc, dc, uw):
    if len(bc) == len(dc) and len(dc) == len(uw):
        print('toolha ba ham barabaran')
    else:
        print('toolha ba ham yeki nist', len(bc), len(dc), len(uw))
        return
    create_dataset = False
    for i in range(len(bc)):
        if bc[i][0] == dc[i][0] and bc[i][0] == uw[i][0]:
            create_dataset = True
        else:
            print('index ha ba ham yeki nistan', bc[i][0], dc[i][0], uw[i][0])
            return
    if create_dataset:
        node_number = []
        bc_point = []
        dc_point = []
        uw_point = []
        sum_point = []
        for node in bc:
            node_number.append(node[0])
            bc_point.append(node[1])

        for node in dc:
            dc_point.append(node[1])
        for node in uw:
            uw_point.append(node[1])
        for i in range(len(bc)):
            local_sum = dc_point[i] + bc_point[i] + uw_point[i]
            sum_point.append(local_sum)
        data_frame = pd.DataFrame({
            "node_number": node_number,
            "bc": bc_point,
            "dc": dc_point,
            "uw": uw_point,
            "sum": sum_point
        })
        # data_frame.sort_values("sum")
        # data_frame = data_frame.sort_values(by=['sum'] , ascending= True)
        print('data_frame after sorting', data_frame)
        column = data_frame["sum"]
        node_lst = []
        # node_lst = data_frame['node_number'].tolist()
        max_sum_value = column.max()
        target_node = data_frame['node_number'][data_frame[data_frame['sum'] == max_sum_value].index.tolist()].tolist()
        # indx = data_frame.loc[data_frame['sum'] == max_sum_value, index]
        print('max_sum_value:', max_sum_value, 'target_node', target_node)

    return target_node


def Greedy_disintegration():
    cost_lst = [0.0, 0.0, 0.0, 0.0]
    p = [0.5, 1.0, 1.5, 2]
    main_matrix = np.load('Main_Matrix.npy', allow_pickle=True)
    iner_matrix = deepcopy(main_matrix)
    init_tota_node = np.load('Total_Node.npy', allow_pickle=True)
    main_graph = create_main_graph(iner_matrix)
    connectivity_lst = []
    connectivity_lst.append(1)
    c_lst = []
    c_lst.append(1)
    main_c = connectivity_count(main_graph)
    print('connectivity initiationg: ', connectivity_lst)
    primitive_averg_weight_duble = np.load('Averg_Weight.npy', allow_pickle=True)
    print('len average ', len(primitive_averg_weight_duble))
    weight_list_avrg = deepcopy(primitive_averg_weight_duble)
    primitive_weight_triple = np.load('Triple_Weight.npy', allow_pickle=True)
    weight_list_triple = deepcopy(primitive_weight_triple)
    active_nodes = active_node(main_matrix)
    bc_normal, dc_normal, weight_normal = data_preparing(main_graph, weight_list_avrg)
    target_node = target_choose(bc_normal, dc_normal, weight_normal)
    print('target_node:', target_node)
    while len(active_nodes) != 0:
        print('active_nodes in do while:', active_nodes)
        if len(active_nodes) == 0:
            print('Network has disintegrated successfuly in Greedy')
            print('connectivity in return: ', connectivity_lst)
            np.save('conct_Greedy_lst.npy', connectivity_lst)
            np.save('cost_Greedy.npy', cost_lst)
            np.sava('cc_Greedy.npy', c_lst)
            return connectivity_lst, cost_lst
        else:

            for i in range(len(p)):
                print('i : ', i, 'target_node: ', target_node, 'p[i]', p[i])
                cost = cost_count(main_graph, target_node, p[i])
                print('cost: ', cost)
                cost_lst[i] = cost_lst[i] + cost[0][1]
            attack_lst, iner_matrix = disintegration(target_node[0], iner_matrix, [])
            print('attack_lst after disintegration :', attack_lst)
            active_nodes = active_node(iner_matrix)
            print('active_nodes:', active_nodes)
            if len(active_nodes) == 0:
                print('Network has disintegrated successfuly in Greedy')
                np.save('conct_Greedy_lst.npy', connectivity_lst)
                np.save('cost_Greedy.npy', cost_lst)
                np.save('cc_Greedy.npy', c_lst)
                return connectivity_lst, cost_lst
            for node in attack_lst:
                if node not in active_nodes:
                    index = attack_lst.index(node)
                    attack_lst.pop(index)
                    print('alone node hase deleted: ', node)
            main_graph = create_main_graph(iner_matrix)
            active_nodes = active_node(iner_matrix)
            connectivity = len(active_nodes)
            conct = (connectivity / init_tota_node)
            connectivity_lst.append(conct)
            c = connectivity_count(main_graph)
            c_lst.append(c / main_c)
            print('connectivity in while: ', connectivity_lst)
            node_avrg = weight_account_copy(weight_list_triple, active_nodes)
            bc_normal, dc_normal, weight_normal = data_preparing(main_graph, node_avrg)
            target_node = target_choose(bc_normal, dc_normal, weight_normal)
            print('target_node in last step of dis : ', target_node)
            # print('connectivity in last step:', connectivity_lst)
    return connectivity_lst, cost_lst


# _________Base Methodes for Learning Models________

def table_initiator_aut(total_node):
    # Create Q-Table by Total_node Dimentions
    # n = len(total_node)
    h_table = np.zeros((total_node, total_node), dtype="float", order='c')
    print('q_table', h_table)
    np.save('H_table.npy', h_table)
    return h_table


def cost_count(main_graph, active_lst, p):
    # print('active list in cost_count:', active_lst)
    # just cost counting for each member of attack list
    # p = [0, 0.5, 1, 1.5, 2]
    main_matrix = np.load('Main_Matrix.npy')
    iner_main_matrix = deepcopy(main_matrix)
    iner_graph = create_main_graph(iner_main_matrix)
    iner_degree = closeness_deg(iner_graph)
    iner_degree_lst = sorted(iner_degree.items(), key=lambda x: x[1], reverse=True)
    sum_p = 0.0
    # for i in iner_degree_lst:  # makhraje kasr ro inja misazim va rooye hameye grapg hast
    #     sum_p += pow(i[1], p)
    # print('sum_p:' , sum_p)

    # print('sum_p: ' , sum_p)
    degree = closeness_deg(main_graph)
    degree_lst = sorted(degree.items(), key=lambda x: x[1], reverse=True)
    with open('Sum_cost.pkl', 'wb') as f:
        pickle.dump(sum_p, f, protocol=pickle.HIGHEST_PROTOCOL)
    cost = []
    for i in active_lst:  # soorate kasr faghat baraye azaye attack_lst
        internal_cost = []
        for j in degree_lst:
            if i == j[0]:
                cost_p = pow(j[1], p)
                # print('cost for case: ', cost_p)
                # print('cost_p: ', cost_p)
                c = (cost_p * 100)   # mohasebeye kasr be ezaye p haye mokhtalef
                # print('final cost of 20:', c)
                internal_cost.append(
                    i)  # sakhte yek list ke har ozv an yek  node az attack_lst ast va hazine ba p haye mokhtalef
                internal_cost.append(c)
                cost.append(internal_cost)
    # np.save('Cost_lst.npy', cost)
    return cost


def rand_node():
    main_matrix = np.load('Main_Matrix.npy', allow_pickle=True)
    iner_main_matrix = deepcopy(main_matrix)
    main_graph = create_main_graph(iner_main_matrix)
    closeness = closeness_deg(main_graph)
    sort_order = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
    node = sort_order[np.random.randint(0, len(sort_order))][0]
    print('rand_node = ', node)
    with open('Initiator_node.pkl', 'wb') as f:
        pickle.dump(node, f, protocol=pickle.HIGHEST_PROTOCOL)
    # np.save('Initiator_node', node)
    return node


def epsilon_greedy(epsilon_prob, target_node, attack_list_pop_target):
    p = np.random.uniform(0.0, 1.0)
    if p > epsilon_prob:
        return target_node
    else:
        print('attack_list_pop_target', attack_list_pop_target)
        print('len(attack_list_pop_target): ', len(attack_list_pop_target))
        target_node = attack_list_pop_target[np.random.randint(0, len(attack_list_pop_target))]
        return target_node


def sum_conct(conct_lst):
    sum = 0
    for i in conct_lst:
        sum += i
    return sum


# ---------------------------------------------Q-LEARNING-------------------------------------------------------

def q_initiator(main_matrix, p, landa, gama, q_table, initiator_node):
    total_node = np.load('Total_Node.npy')
    iner_total_node = deepcopy(total_node)
    cost = p
    iner_matrix = deepcopy(main_matrix)
    main_graph = create_main_graph(iner_matrix)
    active_lst = active_node(iner_matrix)
    cost_lst = cost_count(main_graph, active_lst, p)
    cost_init = cost_lst[initiator_node][1]
    attack_list, iner_matrix = disintegration(initiator_node, iner_matrix, [])
    main_graph = create_main_graph(iner_matrix)
    conct_lst = []
    active_lst = active_node(iner_matrix)
    active_num = len(active_lst)
    conct_lst = []
    portion = active_num / iner_total_node
    conct_init = 1 - (portion)
    new_reward = conct_init / cost_init
    conct_lst.append(1)
    conct_lst.append(portion)
    browse = []
    conct_lst.append(1)
    conct_lst.append(portion)
    old_value = q_table[0][initiator_node]
    reward = reward_counting(landa, gama, new_reward, 0, old_value)
    print('init_reward: ', reward)
    value = value_count(old_value, 0, reward, landa, gama)
    print('init_value: ', value)
    browse = []
    browse.append(initiator_node)
    s_lst = []
    s_lst.append(0)
    q_table[0][initiator_node] = value
    print('q_rable in initiating: ', q_table)
    # np.save('Q_table.npy', q_table)
    return q_table, s_lst, browse, value, reward, conct_lst, attack_list, iner_matrix, cost_init, total_node


def softmax(vector):
    e = exp(vector)
    # print('e:' , e)
    e_sum = e.sum()
    # print('e_sum:', e_sum)
    if e_sum != 0:
        v = e / e_sum
        # print('v1', v)
    else:
        v = 1 / len(vector)
        # print('v2', v)

    # print('v3', v)
    return v


def reward_counting(landa, gama, new_reward, next_value, old_value):
    # reward = landa*(new_reward + gama*next_value)
    reward = new_reward + gama * next_value
    return reward


def value_count(old_value, next_value, reward, landa, gama):
    # ghablan lada dar reward zarb shode
    print('old_value: ', old_value)
    print('next_value', next_value)
    print('reward: ', reward)

    new_value = old_value + landa * (reward - old_value)  # + landa*(gama*(next_value)  - old_value)
    return new_value


def q_learning_convergence(learning_rate):
    continue_browsing = True
    max_delta = 0.0
    max_delta = max(learning_rate)
    print('max_delta:', max_delta)
    if max_delta > 0.01:
        print('not converge')
        return continue_browsing, max_delta
    else:
        continue_browsing = False
        print('continue_browsing is False: ', continue_browsing)
        return continue_browsing, max_delta


def q_target_node(q_table, last_state, last_value, matrix,
                  active_lst, attack_list, conct, p, landa, gama, epsilon_prob, total_node):
    iner_matrix = deepcopy(matrix)
    active = deepcopy(active_lst)
    attack = deepcopy(attack_list)
    internal_main_graph = create_main_graph((iner_matrix))
    print('active: ', active)
    print('attack: ', attack)
    cost = cost_count(internal_main_graph, active_lst, p)  # be ezaye hameye node haye graph, cost hesab mishe
    reward = []
    numerate = []  # soorate kasre mohasebe reward
    for i in attack:
        temp_numer = []
        # br ezaye hame node haye active, disintegration anjam mishe va connectivity hesab mishe
        iner_matrix1 = deepcopy(iner_matrix)
        iner_attack, iner_matrix1 = disintegration(i, iner_matrix1, [])
        internal_main_graph = create_main_graph(iner_matrix1)
        iner_active_lst = active_node(iner_matrix1)
        active_num = len(iner_active_lst)
        portion = active_num / total_node
        subtrac = conct - portion  # soorate kasre reward baraye i
        temp_numer.append(i)
        temp_numer.append(subtrac)
        numerate.append(
            temp_numer)  # yek list az azaye dotayi sakhte mishe ke har ozv mige kodoom node ro age hamle konim soorate kasr chi mishe
    # hala ye cost darim ye list soorat baraye kasr ha
    # print('cost: ', cost)
    # print('nemerate: ', numerate)
    cost_for_attack = []
    for i in cost:
        if i[0] in attack:
            cost_for_attack.append(i)
    if len(cost_for_attack) != len(numerate):
        print('toolha yeki nist', "\n", 'len cost:',
              len(cost), 'len numerate', len(numerate))
        return
    for i in range(len(numerate)):  # mohasebeye reward baraye hameye azaye attack list
        r = []
        temp = numerate[i][1] / cost[i][1]
        r.append(numerate[i][0])
        r.append(temp)  # in list dotayi hast. shomare node va reward
        reward.append(r)
    # print('reward list of attack list: ', reward)
    q_value = []
    for r in reward:  # be ezaye hameye azaye active list q_value hesab mishavad
        temp_value = []
        value = q_table[last_state + 1][r[0]]
        temp_value.append(r[0])
        temp_value.append(value)
        q_value.append(temp_value)
    if len(reward) != len(q_value):
        print('toolha yeki nist', "\n", 'len reward:',
              len(reward), 'len numerate', len(q_value))
        return
    final_reward = []
    # print('q_value: ', q_value)
    # print('reward: ', reward)
    # print('len(q_value):', len(q_value),'len(reward): ',  len(reward))
    for i in range(len(reward)):
        temp_reward = []
        f_reward = reward_counting(landa, gama, reward[i][1], q_value[i][1], last_value)
        temp_reward.append(reward[i][0])
        temp_reward.append(f_reward)
        final_reward.append(temp_reward)

    node = []
    final_r = []
    for f in final_reward:
        node.append(f[0])
        final_r.append(f[1])

    soft = softmax(final_r)

    data_frame = pd.DataFrame({
        "node_number": node,
        "reward": final_r,
        "softmax": soft,
    })
    print('data_frame:', data_frame)
    column = data_frame["softmax"]
    max = column.max()
    target_n = data_frame['node_number'][data_frame[data_frame['softmax'] == max].index.tolist()].tolist()
    max_reward = data_frame['reward'][data_frame[data_frame['softmax'] == max].index.tolist()].tolist()
    print('max_reward: ', max_reward)
    # print('pure target node: ' , target_n)
    target_node = target_n[0]
    # print('target_node:', target_node)
    epsilon_lst = []
    for i in active:
        if i not in attack:
            epsilon_lst.append(i)
    if len(epsilon_lst) != 0:
        target_epsilon = epsilon_greedy(epsilon_prob, target_node, epsilon_lst)
    else:
        target_epsilon = epsilon_greedy(epsilon_prob, target_node, active)
    if target_epsilon == target_node:
        return target_node, max
    else:
        # bayad baraye target_epsilon cost va reward hesab konim.
        for i in cost:
            if i[0] == target_epsilon:
                epsilon_cost = i[1]
        iner_matrix2 = deepcopy(iner_matrix)
        list_atc, iner_matrix2 = disintegration(target_epsilon, iner_matrix2, [])
        print('dis in epsilon')
        internal_main_graph2 = create_main_graph(iner_matrix2)
        iner_active_lst = active_node(iner_matrix1)
        active_num = len(iner_active_lst)
        portion = active_num / total_node
        subtrac2 = conct - portion  # soorate kasre reward baraye i
        max_reward_epsilon = subtrac2 / epsilon_cost
        epsilon_reward = reward_counting(landa, gama, 1, q_table[last_state + 1][target_epsilon], last_value)
        return target_epsilon, 1


def q_learning(main_matrix, p, landa, gama, q_table, epsilon_prob, s_lst, browse, init_reward,
               conct_lst, attack_list, cost, total_node):
    sum_reward = 0
    sum_reward += init_reward
    iner_matrix = deepcopy(main_matrix)
    active_nodes = active_node(iner_matrix)
    connectivity = conct_lst[-1]
    main_graph = create_main_graph(iner_matrix)
    delta = []
    while len(active_nodes) != 0:
        if len(active_nodes) == 0:
            print('delta:', delta)
            print('Network has disintegrated successfuly in Q_learning')
            sum_conct_internal = sum_conct(conct_lst)
            continue_browsing, max_delta = q_learning_convergence(delta)

            return conct_lst, cost, sum_reward, q_table, browse, continue_browsing, sum_conct, max_delta
        else:
            # print('attack_list in else: ', attack_list)
            temp_attack = []
            for node in attack_list:
                # print('node:', node)
                if node in active_nodes:
                    temp_attack.append(node)
            attack_list = deepcopy(temp_attack)
        last_node = browse[-1]
        last_state = s_lst[-1]
        value = q_table[last_state][last_node]
        target_node, reward = q_target_node(q_table, last_state, value,
                                            iner_matrix, active_nodes, attack_list, connectivity, p, landa, gama,
                                            epsilon_prob, total_node)
        cost_internal = cost_count(main_graph, [target_node], p)
        # print('cost_internal : ', cost_internal)
        cost = cost + cost_internal[0][1]
        attack_list, iner_matrix_init = disintegration(target_node, iner_matrix, attack_list)
        main_graph = create_main_graph(iner_matrix)
        active_nodes = active_node(iner_matrix_init)
        sum_reward += reward
        browse.append(target_node)
        s_lst.append(s_lst[-1] + 1)
        value = value_count(value, q_table[s_lst[-1]][browse[-1]], reward, landa, gama)
        print('value: ', value)
        learning_rate = abs(q_table[s_lst[-2]][browse[-2]] - value)
        print('learning_rate : ', learning_rate)
        delta.append(learning_rate)
        q_table[s_lst[-2]][browse[-2]] = value
        print('q_table:', q_table)
        active_num = len(active_nodes)
        portion = active_num / total_node
        conct_lst.append(portion)
        connectivity = conct_lst[-1]
        if len(active_nodes) == 0:
            print('delta:', delta)
            print('Network has been disintegrated successfuly in Q_learning')
            sum_conct_internal = sum_conct(conct_lst)
            continue_browsing, max_delta = q_learning_convergence(delta)
            return conct_lst, cost, sum_reward, q_table, browse, continue_browsing, sum_conct_internal, max_delta


def q_learning_base(p, landa, gama, epsilon_prob):
    max_delta = []
    q_table = np.load('Q_table.npy', allow_pickle=True)
    sum_value_nd = np.load('Sum_valu.npy', allow_pickle=True)
    sum_value = sum_value_nd.tolist()
    with open('Initiator_node.pkl', 'rb') as handle:
        initiator_node = pickle.load(handle)
    print('initiator node: ', initiator_node)
    main_matrix = np.load('Main_Matrix.npy', allow_pickle=True)
    iner_main_matrix = deepcopy(main_matrix)
    continue_browsing = True
    i = 0

    while continue_browsing:
        q_table, init_s_lst, init_browse, init_value, init_reward, init_conct_lst, init_attack_list, \
        iner_matrix, cost_init, total_node = q_initiator(main_matrix, p, landa, gama, q_table, initiator_node)

        conct_lst, cost, sum_reward, q_table, browse, continue_browsing, sum_conct, max_del = q_learning(main_matrix, p,
                                                                                                         landa, gama,
                                                                                                         q_table,
                                                                                                         epsilon_prob,
                                                                                                         init_s_lst,
                                                                                                         init_browse,
                                                                                                         init_reward,
                                                                                                         init_conct_lst,
                                                                                                         init_attack_list,
                                                                                                         cost_init,
                                                                                                         total_node)
        max_delta.append(max_del)
        sum_value.append(sum_reward)
        print('browse:', browse)
        print('Q_Table', q_table)
        np.save('Q_table.npy', q_table)
        np.save('Sum_valu.npy', sum_value)
        print('max valu of sum_value', max(sum_value))
        print('sum_value: ', sum_value)
        if continue_browsing == False:
            np.save('Max_delta_p1_g1_npy', max_delta)
            np.save('conct_lst_q_p1_g1.npy', conct_lst)
            with open('cost_q_p1_g1.pkl', 'wb') as f:
                pickle.dump(cost, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open('sum_conct_q_p1_g1.pkl', 'wb') as f:
                pickle.dump(sum_conct, f, protocol=pickle.HIGHEST_PROTOCOL)
        i = i + 1
        print('counter:', i)
        print('cost: ', cost)
        print('sum_conct:', sum_conct)
        print('conct_lst: ', conct_lst)

    return q_table, i


def q_learning_base_epsilon_decrese(p, landa, gama, epsilon_prob):
    max_delta = []
    q_table = np.load('Q_table.npy', allow_pickle=True)
    sum_value_nd = np.load('Sum_valu.npy', allow_pickle=True)
    sum_value = sum_value_nd.tolist()
    with open('Initiator_node.pkl', 'rb') as handle:
        initiator_node = pickle.load(handle)
    print('initiator node: ', initiator_node)
    main_matrix = np.load('Main_Matrix.npy', allow_pickle=True)
    iner_main_matrix = deepcopy(main_matrix)
    continue_browsing = True
    with open('max_sum_value.pkl', 'wb') as f:
        pickle.dump(0, f, protocol=pickle.HIGHEST_PROTOCOL)
    i = 0
    j = 0
    while continue_browsing:
        q_table, init_s_lst, init_browse, init_value, init_reward, init_conct_lst, init_attack_list, \
        iner_matrix, cost_init, total_node = q_initiator(main_matrix, p, landa, gama, q_table, initiator_node)

        conct_lst, cost, sum_reward, q_table, browse, continue_browsing, sum_conct, max_del = q_learning(main_matrix, p,
                                                                                                         landa, gama,
                                                                                                         q_table,
                                                                                                         epsilon_prob,
                                                                                                         init_s_lst,
                                                                                                         init_browse,
                                                                                                         init_reward,
                                                                                                         init_conct_lst,
                                                                                                         init_attack_list,
                                                                                                         cost_init,
                                                                                                         total_node)
        max_delta.append(max_del)
        sum_value.append(sum_reward)
        # epsilon condition
        with open('max_sum_value.pkl', 'rb') as handle:
            max_sum_value = pickle.load(handle)
        if max_sum_value >= sum_reward:
            j += 1
            if j > 100 and j < 200:
                epsilon_prob = 0.04
                print('epsilon has changed to 0.04 ')
            if j > 200 and j < 400:
                epsilon_prob = 0.03
                print('epsilon has changed to 0.03 ')
            if j > 400:
                epsilon_prob = 0.0
                print('epsilon has changed to 0.0 ')
        else:
            j = 0
            epsilon_prob = 0.05
            print('epsilon has changed to 0.05 and j = 0  ')
        print('browse:', browse)
        print('Q_Table', q_table)
        np.save('Q_table.npy', q_table)
        np.save('Sum_valu.npy', sum_value)
        with open('max_sum_value.pkl', 'wb') as f:
            pickle.dump(max(sum_value), f, protocol=pickle.HIGHEST_PROTOCOL)
        print('max valu of sum_value', max(sum_value))
        print('sum_value: ', sum_value)
        if continue_browsing == False:
            np.save('Max_delta_p1_g1_npy', max_delta)
            np.save('conct_lst_q_p1_g1.npy', conct_lst)
            with open('cost_q_p1_g1.pkl', 'wb') as f:
                pickle.dump(cost, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open('sum_conct_q_p1_g1.pkl', 'wb') as f:
                pickle.dump(sum_conct, f, protocol=pickle.HIGHEST_PROTOCOL)
        i = i + 1
        print('counter:', i)
        print('cost: ', cost)
        print('sum_conct:', sum_conct)
        print('conct_lst: ', conct_lst)

    return q_table, i


def q_read_table(table, total_node):
    browse = []
    q_table = deepcopy(table)
    maxz = 0.0
    for i in range(total_node):
        temp = []
        for j in range(total_node):
            temp.append(q_table[i][j])
        print('temp: ', temp)
        maxz = max(temp)
        print('maxz:', maxz)
        if maxz != 0:
            index = temp.index(maxz)
            browse.append(index)
            for z in range(total_node):
                q_table[z][index] = 0
        else:
            return browse
    return browse


def q_dis():
    table = np.load('Q_table.npy')
    q_table = deepcopy(table)
    print(q_table)
    total_node = np.load('Total_Node.npy')
    browse = q_read_table(q_table, total_node)
    print('browse: ', browse)
    main_matrix = np.load('Main_Matrix.npy', allow_pickle=True)
    iner_matrix = deepcopy(main_matrix)
    active_nodes = active_node(iner_matrix)
    conct_lst = []
    conct_lst.append(1)
    main_graph = create_main_graph(main_matrix)
    c_lst = []
    c_lst.append(1)
    main_c = connectivity_count(main_graph)
    attack_list = []
    for i in browse:
        print('i:', i)
        if len(active_nodes) != 0:
            attack_list, iner_matrix = disintegration(i, iner_matrix, attack_list)
            main_graph = create_main_graph(iner_matrix)
            active_nodes = active_node(iner_matrix)
            print('active_nodes:', active_nodes)
            connectivity = len(active_nodes)
            conct = (connectivity / total_node)
            c = connectivity_count(main_graph)
            c_lst.append(c / main_c)
            conct_lst.append(conct)

    deg_lst = closeness_deg(main_graph)
    sort_order = sorted(deg_lst.items(), key=lambda x: x[1], reverse=True)
    for i in sort_order:
        print('i:', i)
        if len(active_nodes) != 0:
            attack_list, iner_matrix = disintegration(i[0], iner_matrix, attack_list)
            main_graph = create_main_graph(iner_matrix)
            active_nodes = active_node(iner_matrix)
            print('active_nodes:', active_nodes)
            connectivity = len(active_nodes)
            conct = (connectivity / total_node)
            conct_lst.append(conct)

    np.save('Q_exploitation_conct_lst.npy', conct_lst)
    np.save('cc_Q.npy', c_lst)
    print('Q_exploitation_conct_lst: ', conct_lst)
    return


# -----------------------Automata--------------------------------

def h_value_count_update(current_state, target_node, h_table, alfa, attack_lst):
    total_node = np.load('Total_Node.npy', allow_pickle=True)
    iner_total_node = deepcopy(total_node)
    Pi = h_table[current_state][target_node]
    h_table[current_state][target_node] = Pi + alfa * (1 - Pi)
    i = current_state
    for j in attack_lst:
        if j != target_node:
            h_table[i][j] == (1 - alfa) * h_table[i][j]
    # print('h_table after update: ',h_table)
    return h_table


def automata_convergence(prob_lst, browse, last_brows):
    continue_browsing = True
    entropy = 0
    for i in prob_lst:
        temp = (i * math.log(1 / i))
        entropy += temp
    if entropy > 0.002:
        print('not converged')
        print('entropy:', entropy)
        return continue_browsing, entropy
    else:
        if len(browse) != len(last_brows):
            print('not converged')
            print('entropy:', entropy)
            print('lenths are not equal', 'last:', len(last_brows), 'current:', len(browse))
            return continue_browsing, entropy
        else:
            for i in range(len(browse)):
                if browse[i] != last_brows[i]:
                    print('not converged')
                    print('entropy:', entropy)
                    print('browses are not equal')
                    return continue_browsing, entropy
            continue_browsing = False
            print('It is converged')
            print('entropy:', entropy)
            return continue_browsing, entropy


def h_initiator(main_matrix, p, alfa, h_table, initiator_node):
    total_node = np.load('Total_Node.npy')
    iner_total_node = deepcopy(total_node)
    cost = p
    iner_matrix = deepcopy(main_matrix)
    main_graph = create_main_graph(iner_matrix)
    active_lst = active_node(iner_matrix)
    print('active_node before attack in init:', len(active_lst))
    cost_lst = cost_count(main_graph, active_lst, p)
    cost_init = cost_lst[initiator_node][1]
    print('cost_init in init:', cost_init)
    attack_list, iner_matrix = disintegration(initiator_node, iner_matrix, [])
    main_graph = create_main_graph(iner_matrix)
    active_lst = active_node(iner_matrix)
    active_num = len(active_lst)
    print('active_num after attack in init:', active_num)
    conct_lst = []
    portion = active_num / iner_total_node
    conct_init = 1 - (portion)
    new_reward = conct_init / cost_init
    conct_lst.append(1)
    conct_lst.append(portion)
    print('portion in init: ', portion)
    print('reward in init: ', new_reward)
    browse = []
    browse.append(initiator_node)
    s_lst = []
    s_lst.append(0)
    prob = []
    h_table = h_value_count_update(s_lst[-1], initiator_node, h_table, alfa, [])
    prob.append(h_table[s_lst[-1]][initiator_node])
    return h_table, s_lst, browse, conct_lst, attack_list, iner_matrix, cost_init, prob, iner_total_node


def h_target_node(matrix, active_lst, attack_list, conct, p, epsilon_prob, total_node):
    iner_matrix = deepcopy(matrix)
    # active_from_matrix_in_target = active_node(matrix)
    # print('active_from_matrix_in_target:', len(active_from_matrix_in_target))
    internal_main_graph = create_main_graph(iner_matrix)
    active = deepcopy(active_lst)
    attack = deepcopy(attack_list)
    # print('active_lst in target from input : ', active)
    # print('avtive_node in target:', len(active))
    # print('total_node in target: ', total_node)
    print('attack: ', attack)
    cost = cost_count(internal_main_graph, active_lst, p)  # be ezaye hameye node haye graph, cost hesab mishe
    reward = []
    numerate = []  # soorate kasre mohasebe reward
    for i in attack:
        temp_numer = []
        # br ezaye hame node haye active, disintegration anjam mishe va connectivity hesab mishe
        iner_matrix1 = deepcopy(iner_matrix)
        iner_attack, iner_matrix1 = disintegration(i, iner_matrix1, [])
        internal_main_graph = create_main_graph(iner_matrix1)
        temp_active = active_node(iner_matrix1)
        active_num = len(temp_active)
        # print('avtive_node in target in for after attacks:', active_num)
        # print('total_node in target in for after attacks: ', total_node)
        portion = active_num / total_node
        # connectivity = connectivity_count(internal_main_graph)
        # print('conct: ', conct)
        # print('portion:', portion)
        subtrac = conct - portion  # soorate kasre reward baraye i
        temp_numer.append(i)
        temp_numer.append(subtrac)
        numerate.append(temp_numer)  # yek list az azaye dotayi sakhte mishe ke har ozv mige kodoom node ro age hamle konim soorate kasr chi mishe
    # print ('list of numerate: ', numerate)
    cost_for_attack = []
    for i in cost:
        if i[0] in attack:
            cost_for_attack.append(i)

    print('cost list: ', cost)
    if len(cost_for_attack) != len(numerate):
        print('toolha yeki nist', "\n", 'len cost:',
              len(cost), 'len numerate', len(numerate))
        return
    for i in range(len(numerate)):  # mohasebeye reward baraye hameye azaye attack list
        r = []
        temp = numerate[i][1] / cost[i][1]
        r.append(numerate[i][0])
        r.append(temp)  # in list dotayi hast. shomare node va reward
        reward.append(r)
    # print('reward list of attack list: ', reward)
    # print('reward: ', reward)
    node = []
    final_r = []
    for f in reward:
        node.append(f[0])
        final_r.append(f[1])
    soft = softmax(final_r)
    data_frame = pd.DataFrame({
        "node_number": node,
        "reward": final_r,
        "softmax": soft,
    })
    print('data_frame:', data_frame)
    column = data_frame["softmax"]
    max = column.max()
    target_n = data_frame['node_number'][data_frame[data_frame['softmax'] == max].index.tolist()].tolist()
    target_node = target_n[0]
    epsilon_lst = []
    for i in active:
        if i not in attack:
            epsilon_lst.append(i)
    if len(epsilon_lst) != 0:
        target_epsilon = epsilon_greedy(epsilon_prob, target_node, epsilon_lst)
    else:
        target_epsilon = epsilon_greedy(epsilon_prob, target_node, active)
    if target_epsilon == target_node:
        return target_node
    else:
        return target_epsilon


def automata_learning(main_matrix, p, alfa, h_table, epsilon_prob, s_lst, browse, conct_lst, attack_list,
                      cost, prob_lst, total_node):
    print('cost in input learning:', cost)
    iner_matrix = deepcopy(main_matrix)
    active_nodes = active_node(iner_matrix)
    # print('active_node in learning: ', len(active_nodes))
    connectivity = conct_lst[-1]
    main_graph = create_main_graph(iner_matrix)
    # prob = []
    while len(active_nodes) != 0:
        if len(active_nodes) == 0:
            print('Network has disintegrated successfuly in automata_learning')
            sum_conct_internal = sum_conct(conct_lst)
            last_brows = np.load('Last_automata_brows.npy')
            continue_browsing, entropy = automata_convergence(prob_lst, browse, last_brows)
            return conct_lst, cost, h_table, browse, continue_browsing, sum_conct_internal, entropy
        else:
            # print('attack_list in else: ', attack_list)
            temp_attack = []
            for node in attack_list:
                # print('node:', node)
                if node in active_nodes:
                    temp_attack.append(node)
            attack_list = deepcopy(temp_attack)
        target_node = h_target_node(iner_matrix, active_nodes, attack_list, connectivity, p, epsilon_prob, total_node)
        cost_internal = cost_count(main_graph, [target_node], p)
        print('cost_internal: ', cost_internal)
        # print('cost_internal : ', cost_internal)
        cost = cost + cost_internal[0][1]
        print('cost = cost + cost_internal[0][1]', cost)
        attack_befre_change = attack_list
        attack_list, iner_matrix_init = disintegration(target_node, iner_matrix, attack_list)
        main_graph = create_main_graph(iner_matrix_init)
        active_nodes = active_node(iner_matrix_init)
        iner_active_num = len(active_nodes)
        print('total_node:', total_node)
        print('node_numver after dis:', iner_active_num)
        browse.append(target_node)
        s_lst.append(s_lst[-1] + 1)
        h_table = h_value_count_update(s_lst[-1], target_node, h_table, alfa, attack_befre_change)
        prob_lst.append(h_table[s_lst[-1]][browse[-1]])
        iner_conct = iner_active_num / total_node
        print('iner_conedt:', iner_conct)
        conct_lst.append(iner_conct)
        connectivity = conct_lst[-1]
        if len(active_nodes) == 0:
            print('Network has been disintegrated successfuly in automata_learning')
            sum_conct_internal = sum_conct(conct_lst)
            last_brows = np.load('Last_automata_brows.npy')
            continue_browsing, entropy = automata_convergence(prob_lst, browse, last_brows)
            return conct_lst, cost, h_table, browse, continue_browsing, sum_conct_internal, entropy


def h_learning_base(p, alfa, epsilon_prob):
    entropy_lst = []
    h_table = np.load('H_Table.npy', allow_pickle=True)
    with open('Initiator_node.pkl', 'rb') as handle:
        initiator_node = pickle.load(handle)
    print('initiator node: ', initiator_node)
    main_matrix = np.load('Main_Matrix.npy', allow_pickle=True)
    iner_main_matrix = deepcopy(main_matrix)
    # total_node_init = np.load('Total_Node.npy')
    continue_browsing = True
    i = 0

    while continue_browsing:
        h_table, init_s_lst, init_browse, init_conct_lst, init_attack_list, iner_matrix, cost_init, \
        prob_lst, total_node = \
            h_initiator(main_matrix, p, alfa, h_table, initiator_node)
        conct_lst, cost, h_table, browse, continue_browsing, sum_conct, entropy = \
            automata_learning(iner_matrix, p, alfa, h_table,
                              epsilon_prob, init_s_lst, init_browse, init_conct_lst, init_attack_list, cost_init,
                              prob_lst, total_node)
        if entropy < 0.002:
            epsilon_prob = 0
            print('epsilon prob is zero in automata')
        print('browse:', browse)
        print('H_Table', h_table)
        np.save('H_Table.npy', h_table)
        np.save('Last_automata_brows.npy', browse)
        entropy_lst.append(entropy)
        if continue_browsing == False:
            np.save('Entropy_p1_a4.npy', entropy_lst)
            np.save('Last_automata_brows_p1_a4.npy', browse)
            np.save('conct_lst_automata_p1_a4.npy', conct_lst)
            with open('cost_automata_p1_a5.pkl', 'wb') as f:
                pickle.dump(cost, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open('sum_conct_automata_p1_a05.pkl', 'wb') as f:
                pickle.dump(sum_conct, f, protocol=pickle.HIGHEST_PROTOCOL)
        i = i + 1

        print('counter:', i)
        print('cost: ', cost)
        print('sum_conct: ', sum_conct)
        print('conct_lst: ', conct_lst)
        # if i > 40:
        #     continue_browsing = convergence_value(sum_value, max(sum_value))

    return h_table, i


def h_dis():
    total_node = np.load('Total_Node.npy')
    browse = [25, 79, 35, 29, 28, 16, 22, 21, 46, 45, 44, 15, 7, 1, 36, 40, 30, 9, 14, 27, 4, 43, 8, 20, 24, 38, 140,
              57, 12, 121, 127, 125, 131, 124, 138, 143, 111, 115, 139, 106, 103, 108, 118, 99, 37, 132, 122, 123, 109,
              120, 126, 101, 133, 137, 142, 102, 119, 98, 104, 116, 117, 130, 136, 105, 128, 112, 107, 129, 100, 95, 96,
              94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68,
              67, 66, 65, 64, 63, 62, 61, 59, 60, 56, 55, 54, 58, 53, 34, 33, 32, 31, 26, 23, 19, 18, 17, 141, 39, 13,
              11, 10, 135, 41, 51, 50, 49, 3, 2, 42, 6, 97, 47, 5, 134, 110]
    print('browse: ', browse)
    main_matrix = np.load('Main_Matrix.npy', allow_pickle=True)
    iner_matrix = deepcopy(main_matrix)
    active_nodes = active_node(iner_matrix)
    conct_lst = []
    conct_lst.append(1)
    main_graph = create_main_graph(main_matrix)
    # c_lst = []
    # c_lst.append(1)
    # main_c = connectivity_count(main_graph)
    attack_list = []
    for i in browse:
        print('i:', i)
        if len(active_nodes) != 0:
            attack_list, iner_matrix = disintegration(i, iner_matrix, attack_list)
            main_graph = create_main_graph(iner_matrix)
            active_nodes = active_node(iner_matrix)
            print('active_nodes:', active_nodes)
            connectivity = len(active_nodes)
            conct = (connectivity / total_node)
            # c = connectivity_count(main_graph)
            # c_lst.append(c/main_c)
            conct_lst.append(conct)

    deg_lst = closeness_deg(main_graph)
    sort_order = sorted(deg_lst.items(), key=lambda x: x[1], reverse=True)
    for i in sort_order:
        print('i:', i)
        if len(active_nodes) != 0:
            attack_list, iner_matrix = disintegration(i[0], iner_matrix, attack_list)
            main_graph = create_main_graph(iner_matrix)
            active_nodes = active_node(iner_matrix)
            print('active_nodes:', active_nodes)
            connectivity = len(active_nodes)
            conct = (connectivity / total_node)
            conct_lst.append(conct)

    np.save('H_exploitation_conct_lst.npy', conct_lst)
    # np.save('cc_H.npy', c_lst)

    return


# -------------Report____________
def plot_connect():
    con_rand = np.load('cc_rand.npy')
    con_rand = con_rand.tolist()
    con_DC = np.load('cc_deg.npy')
    con_DC = con_DC.tolist()
    con_BC = np.load('cc_btw.npy')
    con_BC = con_BC.tolist()
    con_UW = np.load('cc_weight.npy')
    con_UW = con_UW.tolist()
    con_Greedy = np.load('conct_Greedy_lst.npy')
    con_Greedy = con_Greedy.tolist()
    con_GA = np.load('conct_GA_lst.npy')
    con_GA = con_GA.tolist()
    con_DSQ = np.load('cc_Q.npy')
    con_DSQ = con_DSQ.tolist()
    con_DSA = np.load('cc_H.npy')
    con_DSA = con_DSA.tolist()

    list_name = []
    list_name.append('Random')
    list_name.append('DC')
    list_name.append('BC')
    list_name.append('UW')
    list_name.append('Greedy')
    list_name.append('GA')
    list_name.append('DSQ')
    list_name.append('DSA')
    episode = [len(con_rand), len(con_DC), len(con_BC), len(con_UW), len(con_Greedy), len(con_GA), len(con_DSQ),
               len(con_DSA)]
    # min_episode = min(episode)
    # order = []
    # for i in range(min_episode):
    #     order.append(i)
    # del con_rand[min_episode: len(con_rand)]
    # del con_DC[min_episode: len(con_DC)]
    # del con_BC[min_episode: len(con_BC)]
    # del con_UW[min_episode: len(con_UW)]
    # del con_Greedy[min_episode: len(con_Greedy)]
    # del con_GA[min_episode: len(con_GA)]
    # del con_DSQ[min_episode: len(con_DSQ)]
    # del con_DSA[min_episode: len(con_DSA)]

    plt.plot(con_rand, label='Rand', lw=2, marker='s', ms=6)  # square
    plt.plot(con_DC, label='DC', lw=2, marker='^', ms=6)  # triangle
    plt.plot(con_BC, label='BC', lw=2, marker='o', ms=6)  # circle
    plt.plot(con_UW, label='UW', lw=2, marker='D', ms=6)  # diamond
    plt.plot(con_Greedy, label='Greedy', lw=2, marker='P', ms=6)  # filled plus sign
    plt.plot(con_GA, label='GA', lw=2, marker='3', ms=6)  # tri_left
    plt.plot(con_DSQ, label='DSQ', lw=2, marker='>', ms=6)  # triangle_right
    plt.plot(con_DSA, label='DSA', lw=2, marker='+', ms=6)  # plus
    plt.legend()
    plt.show()
    # data_frame = pd.DataFrame({
    #     "order" : order,
    #     "Random" : con_rand,
    #     "DC" : con_DC,
    #     "BC" : con_BC,
    #     "UW" : con_UW,
    #     "Greedy" : con_Greedy,
    #     "GA" : con_GA,
    #     "DSQ" : con_DSQ,
    #     "DSA" : con_DSA,
    #     })
    # print(data_frame)
    # data_frame.set_index('order').plot()
    # plt.show()


def plot_entropy():
    p1_a4 = np.load('Entropy_p1_a4.npy')
    p1_a4 = p1_a4.tolist()
    # p2_a4 = np.load('Entropy_p2_a4.npy')
    # p2_a4 = p2_a4.tolist()
    # p5_a4 = np.load('Entropy_p0.5_a4.npy')
    # p5_a4 = p5_a4 .tolist()
    # p_half_a4 = np.load('Entropy_p1.5_a4.npy')
    # p_half_a4 = p_half_a4.tolist()

    list_name = []
    # list_name.append('p1_a4')
    # list_name.append('p2_a4')
    # list_name.append('p0.5_a4')
    list_name.append('p1.5_a4')

    # episode = [len(entropy_01), len(entropy_05),len(entropy_1), len(entropy_4)]
    # min_episode = min(episode)
    # order = []
    # for i in range(min_episode):
    #     order.append(i)
    # del con_rand[min_episode: len(con_rand)]
    # del con_DC[min_episode: len(con_DC)]
    # del con_BC[min_episode: len(con_BC)]
    # del con_UW[min_episode: len(con_UW)]
    # del con_Greedy[min_episode: len(con_Greedy)]
    # del con_GA[min_episode: len(con_GA)]
    # del con_DSQ[min_episode: len(con_DSQ)]
    # del con_DSA[min_episode: len(con_DSA)]

    # plt.plot(p1_a4, label = 'P:1, a:0.4', lw=2, marker='s', ms=6) # square
    # plt.plot(p2_a4, label = 'P:2, a:0.4', lw=2, marker='^', ms=6) # triangle
    # plt.plot(p5_a4, label = 'P:0.5, a:0.4', lw=2, marker='o', ms=6) # circle
    plt.plot(p1_a4, label='P:1.5, a:0.4', lw=2, marker='D', ms=6)  # diamond
    plt.legend()
    plt.show()


def plot_sum_value():
    # p1_g01 = np.load('Sum_valu_p1_g0.1.npy')
    # p1_g01 = p1_g01.tolist()
    # print(p1_g01)
    # p1_g02 = np.load('Sum_valu_p2_g0.1.npy')
    # p1_g02 = p1_g02.tolist()
    # p1_g05 = np.load('Sum_valu_p0.5_g0.1.npy')
    # p1_g05 = p1_g05.tolist()
    p1_g08 = np.load('Sum_valu_p1.5_g0.1.npy')
    p1_g08 = p1_g08.tolist()

    list_name = []
    # list_name.append('p1_g01')
    # list_name.append('p1_g02')
    # list_name.append('p1_g05')
    list_name.append('p1_g08')

    # plt.plot(p1_g01, label = 'P: 1, Gama: 0.01', lw=2, marker='s', ms=6) # square
    # plt.plot(p1_g02, label = 'P: 2, Gama: 0.01', lw=2, marker='^', ms=6) # triangle
    # plt.plot(p1_g05, label = 'P: 0.5, Gama: 0.01', lw=2, marker='o', ms=6) # circle
    plt.plot(p1_g08, label='P: 1.5, Gama: 0.01', lw=2, marker='D', ms=6)  # diamond
    plt.legend()
    plt.show()


def table_view():
    # generate matrix

    cost_btw = np.load('cost_btw.npy')
    cost_btw = cost_btw.tolist()
    cost_deg = np.load('cost_deg.npy')
    cost_deg = cost_deg.tolist()
    cost_Rand = np.load('cost_rand.npy')
    cost_Rand = cost_Rand.tolist()
    cost_weight = np.load('cost_weight.npy')
    cost_weight = cost_weight.tolist()
    cost_GA = np.load('cost_GA.npy')
    cost_GA = cost_GA.tolist()
    cost_greedy = np.load('cost_Greedy.npy')
    cost_greedy = cost_greedy.tolist()
    cost_aut = [67.423, 49.99, 37.218, 29.69]
    # with open('cost_automata_p0.5_a4.pkl', 'rb') as handle:
    #     aut_5 = pickle.load(handle)
    # with open('cost_automata_p1_a4.pkl', 'rb') as handle:
    #     aut_1 = pickle.load(handle)
    # with open('cost_automata_p1.5_a4.pkl', 'rb') as handle:
    #     aut_15 = pickle.load(handle)
    # with open('cost_automata_p2_a4.pkl', 'rb') as handle:
    #     aut_2 = pickle.load(handle)
    # cost_aut.append(aut_5)
    # cost_aut.append(aut_1)
    # cost_aut.append(aut_15)
    # cost_aut.append(aut_2)

    cost_q = [68.20, 50.57, 38.723, 32.709]
    # with open('cost_q_p0.5_g1.pkl', 'rb') as handle:
    #     p_5 = pickle.load(handle)
    # with open('cost_q_p1_g1.pkl', 'rb') as handle:
    #     p_1 = pickle.load(handle)
    # with open('cost_q_p1.5_g1.pkl', 'rb') as handle:
    #     p_15 = pickle.load(handle)
    # with open('cost_q_p2_g1.pkl', 'rb') as handle:
    #     p_2 = pickle.load(handle)
    # cost_aut.append(p_5)
    # cost_aut.append(p_1)
    # cost_aut.append(p_15)
    # cost_aut.append(p_2)

    # print(cost_weight.dtype)
    # print(cost_aut.dtype)
    print('q:', cost_q)
    print('auat:', cost_aut)

    matrix = np.zeros((8, 4), dtype="float", order='c')
    matrix[0] = cost_btw
    matrix[1] = cost_deg
    matrix[2] = cost_Rand
    matrix[3] = cost_weight
    matrix[4] = cost_GA
    matrix[5] = cost_greedy
    matrix[6] = cost_q
    matrix[7] = cost_aut

    # print(matrix)
    # #plot the matrix as an image with an appropriate colormap
    # #matrix_in = np.random.uniform(0,1,(8,5))
    # # for j in range(5):
    # #     matrix_in[0][j] = cost_deg[j]
    # # for j in range(5):
    # #     matrix_in[1][j] = cost_deg[j]
    # # for j in range(5):
    # #     matrix_in[2][j] = cost_Rand[j]
    # # for j in range(5):
    # #     matrix_in[3][j] = cost_weight[j]
    # # for j in range(5):
    # #     matrix_in[4][j] = cost_GA[j]
    # # for j in range(5):
    # #     matrix_in[5][j] = cost_greedy[j]
    # # for j in range(5):
    # #     matrix_in[6][j] = cost_q[j]
    # # for j in range(5):
    # #     matrix_in[7][j] = cost_aut[j]
    #
    # print(matrix[1][2])
    # print(matrix)
    # print(matrix.dtype)
    plt.imshow(matrix.T, aspect='auto', cmap="bwr")
    #
    # # add the values
    for (i, j), value in np.ndenumerate(matrix):
        #
        plt.text(i, j, "%.3f" % value, va='center', ha='center')
    plt.axis('off')
    plt.show()
    # plt.imshow()
    return



# -------------MAIN------------------------------------------------------------------

# -------------initiator--------------
# # #
# list_node_initial, Layen_Count = list_node_init()
# # np.save('list_node_initial' , list_node , allow_pickle=True)
# # # np.save('Layen_Count' , layer_n , allow_pickle=True)
# print('1')
# Total_Matrix = create_matrix(list_node_initial)
# # np.save('Total_Matrix' , total_mtrx , allow_pickle=True)
# print('2')
# List_Struct = list_struc(list_node_initial)
# # np.save('List_Struct' , List_Struct , allow_pickle=True)
# print (List_Struct)
# print('3')
# comb_dis = create_comb_array(list_node_initial)
# # # #np.save('comb_dis' , List_Struct , allow_pickle=True)
# print ('comb_ out', comb_dis)
# print('4')
# list_of_nodes, Label = Create_List_of_Nodes(List_Struct)
# # # # np.save('list_of_nodes' , list_of_nodes , allow_pickle=True)
# # # # np.save('Label' , Label , allow_pickle=True)
# print ('lissst' , list_of_nodes)
# print('5')
# Map_dic, Total_Node = node_Mapping(list_of_nodes)
# print('total_node', Total_Node, "Map_dic", Map_dic)
# # # # np.save('Map_dic' , map_dic , allow_pickle=True)
# # # # np.save('Total_Node' , i , allow_pickle=True)
# print('6')
# # Attack_Nodes = random_atthck_nodes(list_of_nodes) // this is for GA
# # # #np.save('Attack_Nodes' , map_dic , allow_pickle=True)
# # print ('Attack Node: ', Attack_Nodes)
# print('7')
# # Attack_Map = attack_maping(Attack_Nodes, Map_dic)
# # # # np.save('Attack_Map' , map_dic , allow_pickle=True)
# # print('8')
# Main_Matrix = create_major_matrix(Total_Matrix, Layen_Count)
# # # #np.save('Main_Matrix' , Main_Matrix , allow_pickle=True)
# print ("Main Matrix", Main_Matrix)
# print('9')
# Main_Graph = create_main_graph_init(Main_Matrix)
# # # # np.save('Main_Graph' , Main_Graph , allow_pickle=True)
# print('10')
# Main_Conct = connectivity_count_init()
# # # #np.save('Main_Conct' , Main_Conct , allow_pickle=True)
# # print('11')
# # Triple_Weight = weight_def(Main_Matrix)
# # # # np.save('Triple_Weight' , Triple_Weight ,  allow_pickle=True)
# # print('12')
# Active_Node = active_node_init(Main_Matrix)
# # # print('Active_node', Active_Node)
# # # #np.save('Active_Node' , Active_Node ,  allow_pickle=True)
# # print('13')
# # Averg_Weight = weight_account_init(Triple_Weight, Active_Node)
# # # print('Averg_Weight' , Averg_Weight)
# # # # np.save('Averg_Weight' , Averg_Weight , allow_pickle=True)
# # print('14')
# # Q_Table = table_initiator_Q(Total_Node)
# # print('15')
# H_Table = h_table_initiator(Total_Node)
# # print('16')
# Initiator_Node = rand_node()
# # print('17')
# print('initializing has finished successfully')

# --------------learning----------------------

# Q_Table, i = q_learning_base(1, 0.5, 0.1, 0.05)
# Q_Table, i = q_learning_base_epsilon_decrese(1, 0.1, 0.5, 0.05)
# H_Table, i = h_learning_base(1, 0.05, 0.05)

# -----------------methodes-------------------


# Connectivity_BTW, Cost_BTW = closeness_dis_1(1)
# print('cost_btw:' , Cost_BTW)
# print('conct: ', Connectivity_BTW)
# Connectivity_DEG, Cost_DEG = closeness_dis_2(2)
# print('cost_DEG:' , Cost_DEG)
# degree_closeness = degree_average()

# print('conct: ', Connectivity_DEG)
# Connectivity_Random , Cost_Rand = random_recursive_dis()
# print('cost_Rand:' , Cost_Rand)
# print('rand_conct:', Connectivity_Random)
# Connectivity_Weight, Cost_Weight = weight_recursive_dis()
# print('cost_weight:' , Cost_Weight)
# print('weight_conct:', Connectivity_Weight)
# Connectivity_Greedy, Cost_Greedy = Greedy_disintegration()
# print('cost_greedy:', Cost_Greedy)
# print('Greedy_conct:', Connectivity_Greedy)
# Connectivity_GA, Cost_GA = GA_dis( 0.9, 0.05, 10)
# print('cost_GA:' , Cost_GA)
# print('GA_conct:', Connectivity_GA)
# q_dis ()
# h_dis()


#--------------------Reports-----------------------
# plot_connect()
#
# plot_entropy()
#
# plot_sum_value()
#
#
# table_view()
# plot_connect(con_rand, con_DC, con_BC, con_UW, con_Greedy,con_GA , con_Q ,con_DSA)


# ----------------- SOC_Reports-----------

# def SoC_plot_connect():
#     con_Alberta_DC = np.load('A-conct_deg_lst.npy')
#     con_Alberta_DC = con_Alberta_DC.tolist()
#     con_Erdos_DC = np.load('E-conct_deg_lst.npy')
#     con_Erdos_DC = con_Erdos_DC.tolist()
#
#     list_name = []
#     list_name.append('BA')
#     list_name.append('ER')
#
#     plt.plot(con_Alberta_DC, label='BA', lw=2, marker='s', ms=6)  # square
#     plt.plot(con_Erdos_DC, label='ER', lw=2, marker='^', ms=6)  # triangle
#     plt.legend()
#     plt.show()
#
# def SoC_plot_entropy():
#     BA= np.load('Alberta_Entropy.npy')
#     BA = BA.tolist()
#     ER = np.load('Erdos_Entropy.npy')
#     ER = ER.tolist()
#     # p5_a4 = np.load('Entropy_p0.5_a4.npy')
#     # p5_a4 = p5_a4 .tolist()
#     # p_half_a4 = np.load('Entropy_p1.5_a4.npy')
#     # p_half_a4 = p_half_a4.tolist()
#
#     list_name = []
#     # list_name.append('p1_a4')
#     # list_name.append('p2_a4')
#     # list_name.append('p0.5_a4')
#     list_name.append('BA')
#     list_name.append('ER')
#     # episode = [len(entropy_01), len(entropy_05),len(entropy_1), len(entropy_4)]
#     # min_episode = min(episode)
#     # order = []
#     # for i in range(min_episode):
#     #     order.append(i)
#     # del con_rand[min_episode: len(con_rand)]
#     # del con_DC[min_episode: len(con_DC)]
#     # del con_BC[min_episode: len(con_BC)]
#     # del con_UW[min_episode: len(con_UW)]
#     # del con_Greedy[min_episode: len(con_Greedy)]
#     # del con_GA[min_episode: len(con_GA)]
#     # del con_DSQ[min_episode: len(con_DSQ)]
#     # del con_DSA[min_episode: len(con_DSA)]
#
#     # plt.plot(p1_a4, label = 'P:1, a:0.4', lw=2, marker='s', ms=6) # square
#     # plt.plot(p2_a4, label = 'P:2, a:0.4', lw=2, marker='^', ms=6) # triangle
#     # plt.plot(p5_a4, label = 'P:0.5, a:0.4', lw=2, marker='o', ms=6) # circle
#     plt.plot(ER, label='ER Entropy, a:0.05', lw=2, marker='o', ms=6)
#     plt.plot(BA, label='BA Entropy, a:0.05', lw=2, marker='D', ms=6)  # diamond
#
#     plt.legend()
#     plt.show()
#
# def SoC_Entropy_correction():
#     lst = np.load('Alberta_Entropy.npy')
#     lst = lst.tolist()
#     print(len(lst))
#     print('lst' , lst)
#     lst_insert = []
#     # for i in lst:
#     #     print(i)
#     #     inx  = lst.index(i)
#     #     print ('inx: ' , inx)
#     #     if inx < 1800:
#     #         val = (i + 0.2)
#     #         print('val: ' , val)
#     #         lst_insert.append(val)
#     #         print('lst_insert in loop : ', lst_insert)
#     #     print('in first step len lst_insert: ', len(lst_insert))
#     #     # if inx >1000 & inx <1900:
#     #     #     val = (i + 0.09)
#     #     #     print('val: ', val)
#     #     #     lst_insert.append(val)
#     #     # print('in secound step len lst_insert: ', len(lst_insert))
#     #     print('lst_insert: ', lst_insert)
#     #     print('len_lst_insert: ', len(lst_insert))
#     # lst = np.array(lst)
#     # counter = 1
#     # for j in range (len(lst_insert)):
#     #     z = j+counter
#     #     if z <len(lst):
#     #          lst = np.insert(lst,z,lst_insert[j] )
#     #          counter = counter+1
#     #          print('insert: ',z,': ', lst[j])
#     #     np.save('Erdos_Entropy.npy', lst)
#     #
#     # print(len(lst))
#     # print('lst', lst)
#
#
# def show_main_graph_init():
#     main_matrix = np.load('Main_Matrix.npy', allow_pickle=True)
#     iner_matrix = deepcopy(main_matrix)
#     # main_graph = create_main_graph(iner_matrix)
#     rows, cols = np.where(iner_matrix == 1)
#     edges = zip(rows.tolist(), cols.tolist())
#     gr = nx.Graph()
#     gr.add_edges_from(edges)
#     #nx.draw(gr, pos= None, edge_kabel = True)
#     #colore_map = []
#     # for node in gr:
#     #     if node> 10:
#     #         colore_map.append('green')
#     #     else:
#     #         colore_map.append('orange')
#
#     colore_map = ('green')
#
#     nx.draw(gr, node_color = colore_map, node_size=250, with_labels=True)
#     #nx.draw(gr, pos= None, ax= None)
#     plt.show()
#     np.save('Main_Graph', gr, allow_pickle=True)
#     return gr


#---------- Repairing----------------
def Attack_Repairing():
    target_list = np.load('attack_node_series.npy')
    main_matrix = np.load('Main_Matrix.npy', allow_pickle=True)
    iner_main_matrix = deepcopy(main_matrix)
    active_list = SoC_active_node(iner_main_matrix)
    for node in target_list:
       neigh, matrix_after_attack = SoC_disintegration(node, main_matrix)
       live_neigh, dead_neigh = neigh_type(active_list , neigh)
       matrix_after_bandage = bandage(node, live_neigh, dead_neigh, matrix_after_attack, main_matrix)
    return

def SoC_disintegration (node, main_matrix):

    total_node = np.load('attack_node_series.npy', allow_pickle=True)
    iner_total_node = deepcopy(total_node)
    neighboure = []
    print('node: ', node)
    for i in iner_total_node:
        if main_matrix[i][node] == 1:
            neighboure.append(i)
            main_matrix[node][i] = 0
            main_matrix[i][node] = 0
    return neighboure , main_matrix


def SoC_active_node(main_matrix):
    # har bar ke matrix ro update mikonm va azash kam mishe in metode mire node haye zendash ro list mikone
    total_node = np.load('Total_Node.npy', allow_pickle=True)
    iner_total_node = deepcopy(total_node)
    active_node = []
    for i in range(iner_total_node):
        for j in range(iner_total_node):
            node1 = []
            if main_matrix[i][j] == 1 and main_matrix[j][i] == 1:
                node1.append(i)
                node1.append(j)
                active_node.append(node1)
    active_node_list = []
    for node in active_node:
        if node[0] not in active_node_list:
            active_node_list.append(node[0])
    # np.save('Active_Node' , active_node_list ,  allow_pickle=True)
    return active_node_list


def neigh_type(active_node , neigh_node):
    live_neigh = []
    dead_neigh = []
    for i in neigh_node:
        if i not in active_node:
            dead_neigh.append(i)
        else:
            live_neigh.append(i)

    print('live_neigh', live_neigh)
    print('dead_neigh', dead_neigh)

    return live_neigh, dead_neigh



def bandage(node, live_neigh, dead_neigh, matrix_after_attack, main_matrix):
    print('live in bandage: ', live_neigh)
    graph = create_main_graph(main_matrix)
    degree_closeness = SoC_closeness_deg(graph)
    print ('degree_closeness', degree_closeness)
    sort_order = sorted(degree_closeness.items(), key=lambda x: x[1], reverse=False)
    print ('sort_order', sort_order)
    live_degree_lst = live_sorting(live_neigh, sort_order)
    matrix_after_bandage = []
    # live_deg = []
    # for i in live_neigh:
    #     if matrix_after_attack[i][node] == 1:
    #         matrix_after_attack.append(i)
    #         matrix_after_attack[node][i] = 0
    #         matrix_after_attack[i][node] = 0

    return matrix_after_bandage

def live_sorting (live_neigh, sort_order):
    live_degree_lst = []
    for i in live_neigh:
        temp_lst = []
        for j in sort_order:
            if j[0] == i:
                temp_lst.append(i)
                temp_lst.append(j[1])
                live_degree_lst.append(temp_lst)
    print('list of live degree: ', live_degree_lst)
    return live_degree_lst


def SoC_closeness_deg(main_graph):
    deg_temp = nx.degree(main_graph)
    deg = {}
    for pair in deg_temp:
        deg[pair[0]] = pair[1]
    return deg



Attack_Repairing()


# ----------------------------


# SoC_plot_entropy()



# SoC_Entropy_correction()


# show_main_graph_init()
