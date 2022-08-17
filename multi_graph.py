# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 00:51:38 2021

@author: Neda
"""


import numpy as np
import random
import itertools
import networkx as nx
from itertools import combinations
from scipy.sparse import csr_matrix
import math
from copy import copy, deepcopy
from sklearn.preprocessing import normalize
import pandas as pd
from operator import itemgetter
import matplotlib.pyplot as plt
#import ga
import pickle
from numpy import exp


def list_node_init():
    """gets layers count as an int variation layer_n
    for each layer creates a random int number of nodes as a member of list_node
    returns list of layer nodes as list_node"""
    layer_n = int(input("input a number as number of layers: "))
    temp = itertools.count(1)
    index = [next(temp) for i in range(layer_n)]
    # print('index:', index)
    list_node = []
    for i in index:
        node = np.random.randint(10, 15)
        list_node.append(node)
    #print('list_node:', list_node)
    np.save('list_node_initial' , list_node , allow_pickle=True)
    np.save('Layen_Count' , layer_n , allow_pickle=True)
    return list_node , layer_n


def list_struc(list_node):
    # gets the number of layers, creates and returns struct
    struc = []
    for x in range(len(list_node)):
        str = [x, list_node[x]]
        struc.append(str)

    #print('struc:', struc)
    print('ggggggggggggg')
    np.save('List_Struct' , struc , allow_pickle=True)
    return struc


def random_weighted_graph(n):
    """" create a random graph by n= nodes number, p = probability , lower and upper wight"""

    #print('in create weighted graph n: ', n)
    p = np.random.uniform(0.1, 0.3)
    #p = 0.1
    z = nx.erdos_renyi_graph(n, p)
    g = nx.gnp_random_graph(n,p)
    m = g.number_of_edges()
    #print('number of edge:', m)
    weights = [np.random.randint(5, 10) for r in range(m)]
    uw_edges = g.edges()
    #print ('edges:', uw_edges)
    adj = np.zeros((n, n), dtype="object", order='c')
    #print('zero_adj', adj)
    for edge in uw_edges:
        #print('edge[0]:', edge[0],' edge[1]', edge[1])
        adj[edge[0]][edge[1]] = 1
        adj[edge[1]][edge[0]] = 1
    # print('zero_adj after:', adj)
    # print('type of g: ', type(g))
    # print(igraph.Graph(uw_edges))
    return  adj#igraph.Graph(uw_edges)


def create_comb_array(list_node):
    strc = list_struc(list_node) # create a list of layer number and number of nodes in each layer
    #print (strc)
    comb = combinations(strc, 2) # create all combinations of struct members
    #print(type(comb))
    #print('comb:', comb)
    comb_array = []
    for p in comb:
        comb_array_temp = []
        comb_array_temp.append(p)
        #print (p)
        B0 = p[0][0]
        B1 = p[0][1]
        B2 = p[1][0]
        B3 = p[1][1]
        comb_array_temp.append(B0)
        comb_array_temp.append(B1)
        comb_array_temp.append(B2)
        comb_array_temp.append(B3)
        R = random.randint(1, ((B1 * B3 // 2))) # random number of edges
        comb_array_temp.append(R)
        comb_array_temp.append(B1+B3)
        comb_array.append(comb_array_temp)
    #print('comb_array:', comb_array)
    np.save('comb_dis' , comb_array , allow_pickle=True)
    return comb_array #for each member of combarray: 0.comb, 1.B0, 2.B1, 3.B2, 4.B3, 5.R(edge_number), 6.n(lenth)


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
    for l in range(n): # n = number of layers
        m = list_node[l]
        adjc_mtrx = random_weighted_graph(m) # craete random weighted graph for each layer. l = number of nodes
        total_mtrx[i][j] = adjc_mtrx
        i = i + 1
        j = j + 1
    """create Bipartite matrixes"""
    comb = create_comb_array(list_node)
    """for each member of combarray: 0.comb, 1.B0, 2.B1, 3.B2, 4.B3, 5.R(edge_number), 6.n(lenth)"""
    for p in comb:
        G = nx.bipartite.gnmk_random_graph(p[2], p[4], p[5])
        G_lenth =(p[6])
        Row_order = range(G_lenth)
        G_adj = nx.bipartite.biadjacency_matrix(G,row_order=Row_order, column_order= Row_order)
        G_array = csr_matrix.toarray(G_adj)
        dens_matrx= np.zeros((G_lenth, G_lenth) , order='c')
        for i in range(G_lenth):
            for j in range(G_lenth):
                dens_matrx[i][j]= int(G_array[i][j])
        for i in range(0 , G_lenth):
            for j in range(0, G_lenth):
                if dens_matrx[i][j]==1:
                    dens_matrx[j][i]= 1
        total_mtrx[p[1]][p[3]] = dens_matrx
        total_mtrx[p[3]][p[1]] = dens_matrx
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
    #print('list of nodes:', list_of_Node)
    for i in range(len(list_of_Node)):
        list_of_Node_lables.append(i)

    np.save('list_of_nodes' , list_of_Node , allow_pickle=True)
    np.save('Label' , list_of_Node_lables , allow_pickle=True)
    print('list_of_nodes:' , list_of_Node)
    return list_of_Node , list_of_Node_lables


def random_atthck_nodes(list_of_nodes):
    #print('len(list_of_nodes):',len(list_of_nodes))

    attacked_number = math.floor(len(list_of_nodes)/4)
    #print('attacken number on nodes:', attacked_number)
    attacked_list = random.sample(list_of_nodes, attacked_number)
    #print('attacked nodes:', attacked_list)
    np.save('Attack_Nodes' , attacked_list , allow_pickle=True)
    return attacked_list


def random_atthck_nodes_GA(list_of_nodes , size ):
    #print('len(list_of_nodes):',len(list_of_nodes))
    if len(list_of_nodes) > size:
        attacked_list = random.sample(list_of_nodes, size)
    else:
        attacked_list = list_of_nodes
    return attacked_list


def attacked_node_struct(attacked_nodes):
    #print('---attacked_nodes which passed to attacked_node_struct:', attacked_nodes)
    ls = np.load('List_Struct.npy' )
    list_struct = deepcopy(ls)
    node_struct = []
    for node in attacked_nodes:
        node_struct_temp = []
        layer = node[0]
        node_num = node[1]# index of attacked node in its layer
        Struct = list_struct[layer]# get struct if the layer that belongs to attacked node
        layer_Node_Number = Struct[1]# getting Node_Number and wide of adjacency matrix
        node_struct_temp.append(layer)
        node_struct_temp.append(node_num)
        node_struct_temp.append(layer_Node_Number)
        node_struct.append(node_struct_temp)
    return node_struct #for each node in node_struct as a p: p[0]=layer, p[1]=node_num, p[2]=layer_lenth


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
        i = i+1
    print('data type of map_dic ', type(map_dic))
    # create a binary pickle file
    with open('Map_dic.pkl', 'wb') as f:
        pickle.dump(map_dic,f, protocol=pickle.HIGHEST_PROTOCOL)
    # np.save('Map_dic' , map_dic , allow_pickle=False)
    np.save('Total_Node', i, allow_pickle=True)
    print('Map_dic:', map_dic)
    print('Total_node:', i)
    return map_dic, i


def create_index_list(i , total_node):
    with open('Map_dic.pkl', 'rb') as handle:
        map_dic = pickle.load(handle)
        print('map_dic', type(map_dic))
    #map_dic = pickle.load('Map_dic.npy', allow_pickle=True)
    iner_map_dic= deepcopy(map_dic)
    print('data type after convert ', type(iner_map_dic))
    print('iner_map_dic', iner_map_dic)
    print(iner_map_dic[0])
    print(len(iner_map_dic))
    index_list = []
    for a in range(total_node):
        temp_node = iner_map_dic[a]
        if temp_node[0]== i:
            index_list.append(a)
    print('index_list' , index_list)
    return index_list


def create_major_matrix(Total_Matrix, Layer_Count):
    # main graph ro misaze
    # Map_dic, Total_Node = node_Mapping(list_of_nodes) / Inha ro darim
    # list_node_initial , Layen_Count = list_node()
    total_node = np.load('Total_Node.npy', allow_pickle= True)
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
                index_list1 = create_index_list(i , iner_total_node)
                index_list2 = create_index_list(j , iner_total_node)
                for mp in index_list2:
                    index_list1.append(mp)
                    bi_index_list = index_list1

                for bi in range(len(bi_index_list)):
                    for ci in range(len(bi_index_list)):
                            if matrix[bi][ci]==1:
                                main_matrix[bi_index_list[bi]][bi_index_list[ci]] = matrix[bi][ci]


    main = np.matrix(np.array(main_matrix))
    np.save('Main_Matrix' , main_matrix , allow_pickle=True)
    return main_matrix


def create_main_graph_init(adjacency_matrix):
    # main graph ro namayesh midim
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    #nx.draw(gr, node_size=500,  with_labels=True)
    #plt.show()
    np.save('Main_Graph' , gr , allow_pickle=True)
    return gr


def create_main_graph(adjacency_matrix):
    # main graph ro namayesh midim
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    #nx.draw(gr, node_size=500,  with_labels=True)
    #plt.show()
    return gr


def create_main_graph_copy(adjacency_matrix):
    # main graph ro namayesh midim
    # for i in range(Total_Node):
    #     for j in range(Total_Node):
    #         if adjacency_matrix[i][j] == 1:
    #             adjacency_matrix[j][i] = 0

    rows , cols = np.where(adjacency_matrix == 1)
    edge = (rows.tolist(),cols.tolist())
    edges = zip(rows.tolist(), cols.tolist())
    #print ('yyyyyyyaaaaalllll',edge)



    gr = nx.Graph()
    gr.add_edges_from(edges)
    #nx.draw(gr, node_size=500,  with_labels=True)
    #plt.show()
    np.save('Main_Graph' , gr , allow_pickle=True)
    return gr


def closeness_btw(main_graph):
    # closeness ha ro mohasebe mikone
    btw = nx.betweenness_centrality(main_graph, normalized= False )
    return btw


def closeness_deg(main_graph):
    deg_temp = nx.degree(main_graph)
    deg = {}
    for pair in deg_temp:
        deg[pair[0]] = pair[1]
    return deg


def attack_Node_Mapping(attack_list):
    # node haye attack ro be shomare haye jadid map mikone
    total_node = np.load('Total_Node.npy', allow_pickle= True)
    iner_total_node = deepcopy(total_node)
    map_dic = np.load('Map_dic.npy', allow_pickle= True)
    iner_map_dic = deepcopy(map_dic)
    index_list = []
    for a in range(iner_total_node):
        temp_node = iner_map_dic[a]
        for node in attack_list:
            if temp_node == node:
                index_list.append(a)
    return index_list


def connectivity_count_init(main_graph):
    connectivity = nx.average_node_connectivity(main_graph)
    np.save('Main_Conct' , connectivity , allow_pickle=True)
    return connectivity


def connectivity_count(main_graph):
    connectivity = nx.average_node_connectivity(main_graph)
    #np.save('Main_Conct' , connectivity , allow_pickle=True)
    return connectivity

def disintegration (node, main_matrix, attack_list):
    # node: the first index of attack node list
    # main_matrix: matrix should updated by each disintegration step
    # attack_List: should update by each disintegration step
    # type = 1: Random / type = 2: DEG / Type = 3: BWN / Type = 4: WGHT
    total_node = np.load('Total_Node.npy', allow_pickle= True)
    iner_total_node = deepcopy(total_node)
    neigh = []
    print('node: ',  node )
    for i in range(iner_total_node):
        if main_matrix[i][node] == 1:
            neigh.append(i)
            main_matrix[node][i] = 0
            main_matrix[i][node] = 0


    for n in neigh:
        if n not in attack_list:
            attack_list.append(n)
    final_attack_list = list(set(attack_list))
    print('diiiiiiiiiiiiiiissssssssssssssssssssss')
    return final_attack_list , main_matrix

def closeness_dis(type):
    # aval bayad ye peygham neshoon bedim ke in che noe disi hast
    cost_lst = [0.0, 0.0, 0.0, 0.0, 0.0]
    p = [0.0, 0.5, 1, 1.5, 2]
    main_matrix = np.load('Main_Matrix.npy', allow_pickle= True)
    main_conct = np.load('Main_Conct.npy', allow_pickle= True)
    iner_main_conct = deepcopy(main_conct)
    iner_matrix = deepcopy(main_matrix)
    main_graph = create_main_graph(iner_matrix)
    connectivity_lst = []
    connectivity_lst.append(1)
    attack_list = []
    switcher={
                1: closeness_btw(main_graph),
                2: closeness_deg(main_graph),
                }
    closeness = switcher.get(type,"Invalid type")
    while len(closeness) != 0:
        switcher={
                1: closeness_btw(main_graph),
                2: closeness_deg(main_graph),
                }
        closeness = switcher.get(type,"Invalid type")
        sort_order = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
        if len(closeness) == 0:
            print('final main matrix for other methodes: ', iner_matrix)
            print ('Network has disintegrated successfuly')
            return connectivity_lst, cost_lst
        else:
            if len(closeness) != 0 and len(attack_list)==0:
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

            attack_list , iner_matrix = disintegration(max_order_node, iner_matrix, attack_list)
            print ('iner_matrix in closeness recursive dis:', "\n", iner_matrix)
            main_graph = create_main_graph(iner_matrix)
            connectivity = connectivity_count(main_graph)
            conct = (connectivity/iner_main_conct)
            connectivity_lst.append(conct)

            print('connectivity_lst', connectivity_lst)


def random_recursive_dis():
     cost_lst = [0.0, 0.0, 0.0, 0.0, 0.0]
     p = [0.0, 0.5, 1.0, 1.5, 2]
     main_matrix = np.load('Main_Matrix.npy', allow_pickle= True)
     iner_matrix = deepcopy(main_matrix)
     main_conct = np.load('Main_Conct.npy', allow_pickle= True)
     iner_main_conct = deepcopy(main_conct)
     main_graph = create_main_graph(iner_matrix)
     connectivity_lst = []
     connectivity_lst.append(1)
     attack_list = []
     closeness = closeness_deg(main_graph)
     while len(closeness) != 0:
        closeness =  closeness_deg(main_graph)
        print('closeness before sorting: ', closeness)
        sort_order = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
        print ('sorted:::', sort_order)
        if len(closeness) == 0:
            print('final main matrix for other methodes: ', iner_matrix)
            print ('Network has disintegrated successfuly')
            return connectivity_lst , cost_lst
        else:
            if len(closeness) != 0 and len(attack_list)==0:
                rand_order_node = sort_order[np.random.randint(0, len(sort_order))][0]
            for node in attack_list:
                if node not in closeness:
                    index = attack_list.index(node)
                    attack_list.pop(index)
                    print('alone node hase deleted: ', node)
            #sort_order , max_order, attack_list_rand = attack_Node_Ordering(attack_list, closeness )

            rand_order_node = sort_order[np.random.randint(0, len(sort_order))][0]

            print('target node: ', rand_order_node)
            for i in range(len(p)):
                cost = cost_count(main_graph, [rand_order_node], p[i])
                cost_lst[i] = cost_lst[i] + cost[0][1]
            attack_list , iner_matrix = disintegration(rand_order_node, iner_matrix, attack_list)
            main_graph = create_main_graph(iner_matrix)
            connectivity = connectivity_count(main_graph)
            conct = (connectivity/iner_main_conct)
            connectivity_lst.append(conct)


def weight_def (main_matrix):
    # be ezaye har node ye vazne tasadofi ijad mikone va liste node haye faal ro ham tashkhis mide va barmigardoone
    list_of_weight = []
    total_node = np.load('Total_Node.npy', allow_pickle= True)
    iner_total_node = deepcopy(total_node)
    for i in range(iner_total_node):
        for j in range(iner_total_node):
            node1 = []
            if main_matrix[i][j] ==1:
                weight = np.random.randint(1, 20)
                node1.append(i)
                node1.append(j)
                node1.append(weight)
                list_of_weight.append(node1)
    #print('list of weight :: ' , list_of_weight)
    # baraye yalha be soorate do tarafe vazn injad shode bood ke too in halgheha yektarafash kardam.
    for node in list_of_weight:
        i = node[0]
        j = node[1]
        for n in list_of_weight:
            if n[0] == j and n[1] == i:
                n[2] = node[2]

    np.save('Triple_Weight' , list_of_weight ,  allow_pickle=True)
    return list_of_weight


def active_node_init(main_matrix):
    #har bar ke matrix ro update mikonm va azash kam mishe in metode mire node haye zendash ro list mikone
    total_node = np.load('Total_Node.npy', allow_pickle= True)
    iner_total_node = deepcopy(total_node)
    active_node =  []
    for i in range(iner_total_node):
        for j in range(iner_total_node):
            node1 = []
            if main_matrix[i][j] ==1 and main_matrix[j][i]==1:
                node1.append(i)
                node1.append(j)
                active_node.append(node1)
    active_node_list = []
    for node in active_node:
        if node[0] not in active_node_list:
            active_node_list.append(node[0])
    np.save('Active_Node' , active_node_list ,  allow_pickle=True)
    return active_node_list


def active_node(main_matrix):
    #har bar ke matrix ro update mikonm va azash kam mishe in metode mire node haye zendash ro list mikone
    total_node = np.load('Total_Node.npy', allow_pickle= True)
    iner_total_node = deepcopy(total_node)
    active_node =  []
    for i in range(iner_total_node):
        for j in range(iner_total_node):
            node1 = []
            if main_matrix[i][j] ==1 and main_matrix[j][i]==1:
                node1.append(i)
                node1.append(j)
                active_node.append(node1)
    active_node_list = []
    for node in active_node:
        if node[0] not in active_node_list:
            active_node_list.append(node[0])
    #np.save('Active_Node' , active_node_list ,  allow_pickle=True)
    return active_node_list


def average_count(list_node):
    # ba estefade az vazne yalhayi ke be node vaslan miyangin migire
    #print ('list_node_out: ', list_node)
    #print('list_node in average_count :', list_node)
    count = 0
    sum = 0
    for node in list_node:
        #print ('Node: ', node)
        count = count+1
        sum = sum + node[2]
    #print('node: ', node, 'count:', count , 'sum:', sum )
    weight_avr = (sum/count)
    #print(sum , '/', count, '=' , weight_avr)

    return weight_avr


def average_count_copy(list_node):
    # ba estefade az vazne yalhayi ke be node vaslan miyangin migire
    #print ('list_node_out: ', list_node)
    #print('list_node in average_count :', list_node)
    count = 0
    sum = 0
    for node in list_node:
        #print ('Node: ', node)
        count = count+1
        sum = sum + node[2]
    #print('node: ', node, 'count:', count , 'sum:', sum )
    weight_avr = (sum/count)
    return weight_avr


def weight_account_init(list_of_weight, active_nodes ):
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
    np.save('Averg_Weight' , node_and_avr_list , allow_pickle=True)
    return node_and_avr_list


def table_initiator_Q(total_node):
    # Create Q-Table_Total by Total_node Dimentions
    #n = len(total_node)
    q_table_total = np.zeros((total_node, total_node), dtype="float", order='c')
    sum_value = []
    np.save('Q_table.npy', q_table_total)
    np.save('Sum_valu.npy', sum_value)

    return q_table_total


def h_table_initiator(total_node):
    # Create H-Table by Total_node Dimentions
    #n = len(total_node)
    h_table = np.zeros((total_node, total_node), dtype="float", order='c')
    sum_valuen = []
    print('h_table', h_table)
    np.save('H_Table.npy' , h_table)
    np.save('Automata_Sum_value.npy', h_table)
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


def weight_account_copy(list_of_weight , active_nodes ):
    # be ezaye list haeye vazn ha average vazn yek node ro hesab mikone va tahesh ham list ro sort mikone
    #print('list_of_weight:::::::::::::' , list_of_weight)
    #print(active_nodes)
    node_and_avr_list = []

    for i in active_nodes:
        list_node_internal = []
        node_and_avr_temp = []
        for node in list_of_weight:
            if i == node[0]:
                list_node_internal.append(node)
        #print('list_node_internal:::::::::::::',list_node_internal)
        node_avr = average_count_copy(list_node_internal)
        node_and_avr_temp.append(node_avr)
        node_and_avr_temp.append(i)
        node_and_avr_list.append(node_and_avr_temp)

    node_and_avr_list.sort()
        #= sorted(node_and_avr_list)
    node_and_avr_list.reverse()
    #print('node_and_avr_list::::::::',node_and_avr_list)
    return node_and_avr_list


def attack_weight_sort(attack_node , node_avrg):
    # list attack ro bar asase vazn sort mikone va khoroojish faghat yek node hast
    #print('attack_nodes in attack weight sort: ', attack_node)
    #print ('node_avrg in attack weight sort: ', node_avrg)
    attack_sort = []
    for node in attack_node:
        internal_node = []
        for nd in node_avrg:
            #print('-------', node , '==============',nd )
            if node == nd[1]:

                internal_node.append(nd[0])
                internal_node.append(node)
                attack_sort.append(internal_node)
                #print('gdfhgsdfhgdfheg' , attack_sort)

    attack_sort.sort()
    attack_sort.reverse()
    #print('attack_sort: ',attack_sort)
    final_sorted_attack_node = []
    for node in attack_sort:
        final_sorted_attack_node.append(node[1])
    #print('final_sorted_attack_node: ', final_sorted_attack_node)
    return final_sorted_attack_node


def attack_weight_sort_copy(attack_node , node_avrg):
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
    #print('attack_sort: ',attack_sort)
    final_sorted_attack_node = []
    for node in attack_sort:
        final_sorted_attack_node.append(node[1])
    #print('final_sorted_attack_node: ', final_sorted_attack_node)
    return final_sorted_attack_node


def weight_recursive_dis():
     # recursive disintegration ro anjam mide
     #iner_main_matrix = [row[:] for row in main_matrix]
     cost_lst = [0.0, 0.0, 0.0, 0.0, 0.0]
     p = [0.0, 0.5, 1.0, 1.5, 2]
     main_matrix = np.load('Main_Matrix.npy', allow_pickle= True)
     iner_main_matrix = deepcopy(main_matrix)
     main_conct = np.load('Main_Conct.npy', allow_pickle= True)
     iner_main_conct = deepcopy(main_conct)
     main_graph = create_main_graph(iner_main_matrix)
     connectivity_lst = []
     connectivity_lst.append(1)
     attack_list = []
     avrag_weight = np.load('Averg_Weight.npy', allow_pickle= True)
     iner_averg_weigh = deepcopy(avrag_weight)
     print('iner_averg_weigh' , iner_averg_weigh)

     active_nodes = active_node(iner_main_matrix)
     while len(active_nodes) != 0:
        for node in attack_list:
            if node not in active_nodes:
                index = attack_list.index(node)
                attack_list.pop(index)
                print('alone node hase deleted: ', node)
        if len(active_nodes)!= 0 and len(attack_list)==0:

                # baraye jologiri az khata ya loop binahayat da moredi ke graph az aval chand bakhshi boode
                main_graph = create_main_graph(iner_main_matrix)
                triple_weight = np.load('Triple_Weight.npy', allow_pickle= True)
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
                attack_list , iner_main_matrix = disintegration(target_node, iner_main_matrix, attack_list)
                main_graph = create_main_graph(iner_main_matrix)
                connectivity = connectivity_count(main_graph)
                conct = (connectivity/iner_main_conct)
                connectivity_lst.append(conct)
                #print('attack_list', attack_list)
                #list_of_weight  = weight_def (iner_main_matrix)
                active_nodes = active_node(iner_main_matrix)
                node_averg = weight_account(list_of_weight, active_nodes)
                attack_list = attack_weight_sort(attack_list , node_averg)
     if len(active_nodes) == 0:
            print ('Network has disintegrated successfuly by wight method ')
            return connectivity_lst , cost_lst


def attack_maping(attack_list, map_dic):
    attack_map = []
    for node in attack_list:
        for n in map_dic:
            if node == map_dic[n]:
                attack_map.append(n)
    np.save('Attack_Map' , attack_map , allow_pickle=True)
    print('attack_map::::::::::', attack_map)
    return attack_map


def normalize(abnormal_list):
    ab_normal_list = deepcopy(abnormal_list)
    # list ha jofti hastan. aval bayad yek ozvishoon kinim bad aza ro normal konim, bad dobare set konim.
    # nokteye mohem ine li nabayad az index 1 aza be onvan index join shodan estefade konim chon ina shomare node ha
    # hastan va daem dar hale taghir.
    #print ('abnormal list',ab_normal_list)
    if len(ab_normal_list)!=0:
        node = []
        point = []
        for n in ab_normal_list:
            node.append(n[0])
            point.append(n[1])
        #print ('point' , point)
        #print ('Node: ', node)
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
        #print('Normalized List:',norm_list)
        normal_list_final = []
        for i in range(len(ab_normal_list)):
            internal_point = []
            internal_point.append(node[i])
            internal_point.append(norm_list[i])
            normal_list_final.append(internal_point)
        #print('normal_list_final', normal_list_final)
        return normal_list_final
    else:
        return []


#------------------ GA-------------

def fitness_count(Averg_Weight, main_graph , initiator ):

    weight_list_avrg = deepcopy(Averg_Weight)

    bc = closeness_btw(main_graph)
    dc = closeness_deg(main_graph)
    bc_sort = sorted(bc.items(), key=lambda x: x[1], reverse=True)
    dc_sort = sorted(dc.items(), key=lambda x: x[1], reverse=True)
    #weight_list_avrg = weight_account_copy(weight_list_triple , attack_list)
    print('weight: ', weight_list_avrg, "\n",  'bc: ', bc_sort ,"\n",  'dc', dc_sort)

    weight_list_reverse = []
    for n in weight_list_avrg:
        temp_n = []
        temp_n.append(n[1])
        temp_n.append(n[0])
        weight_list_reverse.append(temp_n)
    print('weight_list_reverse' , weight_list_reverse)
    weight_normal = normalize(weight_list_reverse)
    bc_normal = normalize(bc_sort)
    dc_normal = normalize(dc_sort)
    #motabeghe node haye attack az bd , dc , weight mikeshe biron
    weight_lst = []
    bc_lst = []
    dc_lst = []
    #create attack list from bc dc uw
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
    print('methode: fitness_count: ', 'len weight_lst:', len(weight_lst), 'len dc_lst:', len(dc_lst),'len bc_lst: ', len(bc_lst))
    print('len in fitness_count:')
    print('len(weight_lst):', len(weight_lst), "\n", 'len(dc_lst)', len(dc_lst), "\n", 'len(bc_lst)', len(bc_lst))
    return weight_lst, dc_lst, bc_lst


def fitness_arrenge(uw, dc, bc, initiator):
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
        print('i:', i)
        print('bc_int[i][0] : ', bc_int[i][0])
        print('dc_int[i][0] : ', dc_int[i][0])
        print('uw_int[i][0] : ', uw_int[i][0])
        print('bc len: ', len(bc_int))
        print('dc len: ', len(dc_int))
        print('uw len: ', len(uw_int))
        if bc_int[i][0] == dc_int[i][0] and bc_int[i][0] == uw_int[i][0]:
            create_dataset = True
        # else:
        #     print('index ha ba ham yeki nistan', bc_int[i][0], dc_int[i][0],  uw[i][0])
        #     for node in uw:
        #         if node[i][0] == uw[i][0]:
        #             index = uw.index(node)
        #             uw.pop(index)
    if len(bc_int)== len(dc_int) and len(dc_int) == len(uw_int):
        print('toolha ba ham barabaran')
    else:
        print('toolha ba ham yeki nist', len(bc_int), len(dc_int) , len(uw_int))
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
            "node_number" : node_number,
            "bc" : bc_point,
            "dc" : dc_point,
            "uw" : uw_point,
            "sum" : sum_point
        })
        #data_frame.sort_values("sum")
        data_frame = data_frame.sort_values(by=['sum'] , ascending= True)
        print ('data_frame after sorting', data_frame)
        column = data_frame["sum"]
        node_lst = []
        node_lst = data_frame['node_number'].tolist()
        max_sum_value = column.max()
        target_node = data_frame['node_number'][data_frame[data_frame['sum'] == max_sum_value].index.tolist()].tolist()
        #indx = data_frame.loc[data_frame['sum'] == max_sum_value, index]
        print('max_sum_value:' , max_sum_value , 'target_node' , target_node)
        print(type(node_lst))
        print('node_lst' , node_lst)
        return node_lst


def split_initial_lst(initiate_lst, initial_len,  mutation_portion, crossover_portion):
    print('initiated list in split:' , initiate_lst)
    print('type:', type(initiate_lst))
    iner_init_cross_lst = deepcopy(initiate_lst)

    #portion counting
    mutation_count = math.ceil(mutation_portion*initial_len)
    cross_count = (math.ceil(crossover_portion* (initial_len-mutation_count))-1)
    parent_count = initial_len - mutation_count-cross_count
    print('mutation_count:' , mutation_count)
    print('cross_count:' ,cross_count)
    print('parent_count:', parent_count)

    #create lists
    mut_cross_lst = []
    for i in range(mutation_count):
        mut_cross_lst.append(initiate_lst[i])

    # create cross members after pop mut and befor pop parent
    for node in mut_cross_lst:
            if node in iner_init_cross_lst:
                index = iner_init_cross_lst.index(node)
                iner_init_cross_lst.pop(index)
    print('base: ', iner_init_cross_lst)

    # select cross
    children = random.sample(iner_init_cross_lst, cross_count)
    #children = random.randint(iner_init_cross_lst, cross_count)
    parent = []
    for i in iner_init_cross_lst:
        if i not in children:
            parent.append(i)
    parent_final = random.sample(parent , parent_count)
    #parent_final = random.randint(parent , parent_count)

    print('mut_cross_lst:' , mut_cross_lst , 'parent_final:', parent_final , 'children:' , children)


    return mut_cross_lst , parent_final , children


def split_crossover_lst(crossover_lst_sorted, crossover_lst_len, children_len, mutation_portion, crossover_portion):

    print('the ininiate list of children for crossover in crossover method:', crossover_lst_sorted)
    iner_init_cross_lst = deepcopy(crossover_lst_sorted)
    #children counting
    mutation_count = math.ceil(mutation_portion*children_len)
    cross_count = math.ceil(crossover_portion* (children_len-mutation_count))
    parent_count = children_len - mutation_count-cross_count
    print('mutation_count:' , mutation_count , 'cross_count:' , cross_count, 'parent_count:', parent_count)

    #mut len counting in cross
    mut_cross_len =math.ceil( mutation_portion* crossover_lst_len)


    # create seperatedl list of cross_lst_sorted
    mut_cross_lst = []
    for i in range(mut_cross_len):
        mut_cross_lst.append(crossover_lst_sorted[i])

    # create cross members after pop mut and befor pop parent
    for node in mut_cross_lst:
            if node in iner_init_cross_lst:
                index = iner_init_cross_lst.index(node)
                iner_init_cross_lst.pop(index)

    # parent and child len counting on cross
    cross_cross_len = math.ceil(crossover_portion * len(iner_init_cross_lst))
    parent_cross_len = len(iner_init_cross_lst) - cross_cross_len
    print('mut_cross_len:', mut_cross_len ,"\n",  'cross_cross_len:', cross_cross_len,"\n",  'parent_cross_len:', parent_cross_len)

    # cross_cross_lst = []
    # for i in range(cross_cross_len):
    #     cross_cross_lst.append(crossover_lst_sorted[i+mut_cross_len-1])

    # select cross
    print('mut_cross_lst:' , mut_cross_lst,"\n",  'mutation_count:' , mutation_count)
    print('iner_init_cross_lst:' ,iner_init_cross_lst, "\n", 'cross_count:', cross_count)
    print('parent_cross_len:' , parent_cross_len)
    mut_final = random.sample(mut_cross_lst, mutation_count)
    children = random.sample(iner_init_cross_lst, cross_count)
    parent = []
    for i in iner_init_cross_lst:
        if i not in children:
            parent.append(i)
    parent_final = random.sample(parent, parent_count)

    print('mut_final:' , mut_final , 'parent_final:', parent_final , 'children:' , children)
    return mut_final , parent_final , children


def crossover(children, main_martix, last_gen):
    total_node = np.load('Total_Node.npy', allow_pickle= True)
    # for num in children:
    #     num = int(num)
    print(children)
    iner_total_node = deepcopy(total_node)
    neigh = []
    neigh_single = []
    neigh_final = []
    for i in children:
        print('i in children: ', i)
        for j in range(iner_total_node):
            print('j in range(iner_total_node) : ', j)
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
    print('neighhhhhhhhhhhhhhhh:', neigh )
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
    generation_size = len (initiator)
    print('methode: list_initiate: ','len initiate list: ', len(initiator), "\n",  'initiate list:' , initiator)
    return initiator, generation_size


def list_sorting( weight_Triple, main_matrix, main_graph, initiate_lst):
    iner_active_node = []
    sorted_lst = []
    sorted_lst_int = []
    iner_active_node = active_node(main_matrix)
    print('len(iner_active_node): ', len(iner_active_node))
    if len(iner_active_node)< 3:
        sorted_lst = deepcopy(iner_active_node)
        for num in sorted_lst:
            num = int(num)
            sorted_lst_int.append(num)
        print('****************** node ha tamoom shodan ********************')
        return sorted_lst_int
    Averg_Weight = weight_account(weight_Triple, iner_active_node)
    weight_lst, dc_lst, bc_lst = fitness_count(Averg_Weight,main_graph , initiate_lst )
    #bc , dc, uw = list_allignment(bc_lst, dc_lst, weight_lst)
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


def GA_target_node(mutation_portion , crossover_portion, initiate_lst, generation_size, main_graph, main_matrix, evolution,
               weight_average, weight_triple ):
    iner_init = deepcopy(initiate_lst)
    if len(iner_init)<5:
        print('list is shorter than generation')
        target_node = iner_init[-1]
        return target_node

    iner_matrix = deepcopy(main_matrix)
    print('sorted_lst in GA_target_node:', initiate_lst)
    print('iner_init: ', iner_init)
    print('sorted_lst type: ', type(initiate_lst))
    print('evolution: ', evolution)
    print('generation_size:' , generation_size)
    i = 0
    while i < evolution:
        if len(iner_init) < generation_size:
            print('list is shorter than generation')
            target_node = iner_init[-1]
            return target_node
        # gen avaliya ro dorost mikone
        last_gen = list_sorting(weight_triple, iner_matrix, main_graph , iner_init)
        if len(last_gen) < 3:
            target_node = last_gen[-1]
            return target_node
        print('last_gen in first step:', last_gen)
        target_node = last_gen[-1]
        print('target_node on first step:', target_node)
        last_gen_len = len(last_gen)
        mut , permanent_parent_lst , children = split_initial_lst(last_gen, generation_size,  mutation_portion, crossover_portion)
        print('children in target node: ', children)
        children_len = len(children)
        crossover_lst = crossover(children, iner_matrix, last_gen)
        print('crossover before split in target_node methode: ', crossover_lst)
        crossover_lst_sorted = list_sorting(weight_triple,iner_matrix, main_graph , crossover_lst)
        crossover_lst_len = len(crossover_lst)
        new_mut,new_per_par, new_children = split_crossover_lst(crossover_lst_sorted, crossover_lst_len, children_len, mutation_portion, crossover_portion)
        print('iner_init:' , iner_init)
        print('mut:', mut, "\n", 'new_mut:',  new_mut, "\n", 'permanent_parent_lst:', permanent_parent_lst,"\n",
              'new_per_par', new_per_par, "\n", 'new_children:' ,new_children)
        generation = list_constructor(mut, new_mut, permanent_parent_lst, new_per_par, new_children)
        print('generation before while', initiate_lst)
        print('generation after generation:', generation)
        last_gen = generation
        target_node = generation[-1]
        print('target_node after generation, second step: ', target_node)
        i = i+1
        print('i::::',  i)
    return target_node


def GA_dis( crossover, mutation_portion, evolution):
    cost_lst = [0.0, 0.0, 0.0, 0.0, 0.0]
    p = [0.0, 0.5, 1.0, 1.5, 2]
    #initiator = deepcopy(primitive)
    #init_len = len(initiator)
    main_matrix = np.load('Main_Matrix.npy', allow_pickle= True)
    iner_matrix = deepcopy(main_matrix)
    main_conct = np.load('Main_Conct.npy', allow_pickle= True)
    Averg_Weight = np.load('Averg_Weight.npy' , allow_pickle= True)
    Triple_Weight = np.load('Triple_Weight.npy', allow_pickle=True)
    iner_main_conct = deepcopy(main_conct)
    main_graph = create_main_graph(iner_matrix)
    connectivity_lst = []
    connectivity_lst.append(1)
    active_nodes = active_node(iner_matrix)
    while len(active_nodes) != 0:

        if len(active_nodes) == 0:
            print ('Network has disintegrated successfuly in GA')
            return connectivity_lst, cost_lst

        initiate_lst , generation_size = list_initiate(iner_matrix)
        if len(initiate_lst) == 0:
            print ('Network has disintegrated successfuly in GA')
            return connectivity_lst, cost_lst

        for node in initiate_lst:
            if node not in active_nodes:
                index = initiate_lst.index(node)
                initiate_lst.pop(index)
                print('alone node hase deleted: ', node)
        if len(initiate_lst)<5  :
            print('list is shorter than generation')
            target_node =initiate_lst[-1]
            print('target_node in GA_dis:', target_node)
            for i in range(len(p)):
                cost = cost_count(main_graph, [target_node], p[i])
                cost_lst[i] = cost_lst[i] + cost[0][1]
            attack_lst, iner_matrix = disintegration(target_node, iner_matrix, [])
            active_nodes = active_node(iner_matrix)
            print('active_node after dis:', active_nodes)
            main_graph = create_main_graph(iner_matrix)
            connectivity = connectivity_count(main_graph)
            conct = (connectivity/iner_main_conct)
            connectivity_lst.append(conct)
            if len(active_nodes)== 0:
                print ('Network has disintegrated successfuly in GA')
                return connectivity_lst, cost_lst
        else:

            target_node = GA_target_node(mutation_portion , crossover, initiate_lst, generation_size , main_graph, iner_matrix, evolution,
               Averg_Weight, Triple_Weight )
            print('target_node in GA_dis:', target_node)
            for i in range(len(p)):
                cost = cost_count(main_graph, [target_node], p[i])
                cost_lst[i] = cost_lst[i] + cost[0][1]
            attack_lst, iner_matrix = disintegration(target_node, iner_matrix, [])
            active_nodes = active_node(iner_matrix)
            print('active_node after dis:', active_nodes)
            main_graph = create_main_graph(iner_matrix)
            connectivity = connectivity_count(main_graph)
            conct = (connectivity/iner_main_conct)
            connectivity_lst.append(conct)
            if len(active_nodes)== 0:
                print ('Network has disintegrated successfuly in GA')
                return connectivity_lst, cost_lst
    return connectivity_lst, cost_lst


#___________________Greedy_____________
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
    print('weight_list_reverse' , weight_list_reverse)
    weight_list_reverse_sort = sorted(weight_list_reverse, key=itemgetter(0))
    print('weight_list_reverse_sort:' , weight_list_reverse_sort)
    weight_normal = normalize(weight_list_reverse_sort)
    bc_normal = normalize(bc_sort)
    dc_normal = normalize(dc_sort)
    bc_normal = sorted(bc_normal, key=itemgetter(0))
    dc_normal = sorted(dc_normal, key=itemgetter(0))
    print('bc_normal: ', bc_normal, "\n",  'dc_normal: ', dc_normal, "\n",  'weight_normal: ', weight_normal)
    print(len(bc_normal), len(dc_normal) , len(weight_normal))
    return bc_normal, dc_normal, weight_normal


def target_choose(bc, dc, uw):

    if len(bc)== len(dc) and len(dc) == len(uw):
        print('toolha ba ham barabaran')
    else:
        print('toolha ba ham yeki nist', len(bc), len(dc) , len(uw))
        return
    create_dataset = False
    for i in range(len(bc)):
        if bc[i][0] == dc[i][0] and bc[i][0] == uw[i][0]:
            create_dataset = True
        else:
            print('index ha ba ham yeki nistan', bc[i][0], dc[i][0],  uw[i][0])
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
            "node_number" : node_number,
            "bc" : bc_point,
            "dc" : dc_point,
            "uw" : uw_point,
            "sum" : sum_point
        })
        #data_frame.sort_values("sum")
        #data_frame = data_frame.sort_values(by=['sum'] , ascending= True)
        print ('data_frame after sorting', data_frame)
        column = data_frame["sum"]
        node_lst = []
        #node_lst = data_frame['node_number'].tolist()
        max_sum_value = column.max()
        target_node = data_frame['node_number'][data_frame[data_frame['sum'] == max_sum_value].index.tolist()].tolist()
        #indx = data_frame.loc[data_frame['sum'] == max_sum_value, index]
        print('max_sum_value:' , max_sum_value , 'target_node' , target_node)

    return target_node


def Greedy_disintegration():
    cost_lst = [0.0, 0.0, 0.0, 0.0, 0.0]
    p = [0.0, 0.5, 1.0, 1.5, 2]
    main_matrix = np.load('Main_Matrix.npy', allow_pickle= True)
    iner_matrix = deepcopy(main_matrix)
    main_conct = np.load('Main_Conct.npy', allow_pickle= True)
    iner_main_conct = deepcopy(main_conct)
    main_graph = create_main_graph(iner_matrix)
    connectivity_lst = []
    connectivity_lst.append(1)
    print('connectivity initiationg: ', connectivity_lst)
    primitive_averg_weight_duble = np.load('Averg_Weight.npy', allow_pickle= True )
    print ('len average ', len(primitive_averg_weight_duble))
    weight_list_avrg = deepcopy(primitive_averg_weight_duble)
    primitive_weight_triple = np.load('Triple_Weight.npy', allow_pickle= True )
    weight_list_triple = deepcopy(primitive_weight_triple)
    active_nodes = active_node(main_matrix)
    bc_normal, dc_normal, weight_normal = data_preparing(main_graph, weight_list_avrg)
    target_node = target_choose(bc_normal, dc_normal, weight_normal)
    print('target_node:', target_node)
    while len(active_nodes) != 0:
         print('active_nodes in do while:',active_nodes)
         if len(active_nodes) == 0:
             print('Network has disintegrated successfuly in Greedy')
             print('connectivity in return: ', connectivity_lst)
             return connectivity_lst, cost_lst
         else:

             for i in range(len(p)):
                print('i : ', i , 'target_node: ' , target_node, 'p[i]',  p[i])
                cost = cost_count(main_graph, target_node, p[i])
                print('cost: ' , cost)
                cost_lst[i] = cost_lst[i] + cost[0][1]
             attack_lst, iner_matrix = disintegration(target_node[0], iner_matrix, [])
             print('attack_lst after disintegration :', attack_lst)
             active_nodes = active_node(iner_matrix)
             if len(active_nodes)== 0:
                 print ('Network has disintegrated successfuly in Greedy')
                 return connectivity_lst, cost_lst
             for node in attack_lst:
                if node not in active_nodes:
                    index = attack_lst.index(node)
                    attack_lst.pop(index)
                    print('alone node hase deleted: ', node)
             main_graph = create_main_graph(iner_matrix)
             connectivity = connectivity_count(main_graph)
             conct = (connectivity/iner_main_conct)
             connectivity_lst.append(conct)
             print('connectivity in while: ', connectivity_lst)
             node_avrg = weight_account_copy(weight_list_triple, active_nodes)
             bc_normal, dc_normal, weight_normal = data_preparing(main_graph, node_avrg)
             target_node = target_choose(bc_normal, dc_normal, weight_normal)
             print('target_node in last step of dis : ',target_node)
             print('connectivity in last step:', connectivity_lst)
    return connectivity_lst, cost_lst


#_________Base Methodes for Learning Models________

def table_initiator_aut(total_node):
    # Create Q-Table by Total_node Dimentions
    #n = len(total_node)
    h_table = np.zeros((total_node, total_node), dtype="float", order='c')
    print('q_table' , h_table)
    np.save('H_table.npy' , h_table)
    return h_table


def cost_count(main_graph , active_lst , p):
    #print('active list in cost_count:', active_lst)
    # just cost counting for each member of attack list
    #p = [0, 0.5, 1, 1.5, 2]
    degree = closeness_deg(main_graph)
    degree_lst = sorted(degree.items(), key=lambda x: x[1], reverse=True)
    #print('degree_lst: ', degree_lst)
    sum_p = 0.0
    for i in degree_lst: # makhraje kasr ro inja misazim va rooye hameye grapg hast
        sum_p = sum_p + pow(i[1],p)
    #print('sum_p: ' , sum_p)
    cost = []
    for i in active_lst: # soorate kasr faghat baraye azaye attack_lst
      internal_cost = []
      #print('i: ' , i )
      for j in degree_lst:
       #print('j:' , j)
        #print(j[0])
        if i == j[0]:

            cost_p = j[1]**p
            #print('cost_p: ', cost_p)
            c = (cost_p/sum_p)*100 # mohasebeye kasr be ezaye p haye mokhtalef
            internal_cost.append(i) # sakhte yek list ke har ozv an yek  node az attack_lst ast va hazine ba p haye mokhtalef
            internal_cost.append(c)
            cost.append(internal_cost)


    #print('cost::::' , cost)

    return cost


def rand_node():
    main_matrix = np.load('Main_Matrix.npy' , allow_pickle= True)
    iner_main_matrix = deepcopy(main_matrix)
    main_graph = create_main_graph(iner_main_matrix)
    closeness = closeness_deg(main_graph)
    sort_order = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
    node = sort_order[np.random.randint(0, len(sort_order))][0]
    print('rand_node = ', node)
    with open('Initiator_node.pkl', 'wb') as f:
        pickle.dump(node,f, protocol=pickle.HIGHEST_PROTOCOL)
    #np.save('Initiator_node', node)
    return node


def epsilon_greedy(epsilon_prob, target_node, attack_list_pop_target):
    p = np.random.uniform(0.0, 1.0)
    if p > epsilon_prob:
        return target_node
    else:
        print('attack_list_pop_target', attack_list_pop_target)
        print('len(attack_list_pop_target): ', len(attack_list_pop_target))
        target_node = attack_list_pop_target [np.random.randint(0, len(attack_list_pop_target))]
        return target_node




#---------------------------------------------Q-LEARNING-------------------------------------------------------

def q_initiator(main_matrix, p, landa, gama, q_table, initiator_node):
    main_conct = np.load('Main_Conct.npy')
    iner_main_conct = deepcopy(main_conct)
    cost = p
    iner_matrix = deepcopy(main_matrix)
    main_graph = create_main_graph(iner_matrix)
    active_lst = active_node(iner_matrix)
    cost_lst = cost_count(main_graph, active_lst, p)
    cost_init = cost_lst[initiator_node][1]
    attack_list, iner_matrix= disintegration(initiator_node, iner_matrix, [])
    main_graph = create_main_graph(iner_matrix)
    conct_lst = []
    connectivity = connectivity_count(main_graph)
    conct_init = iner_main_conct - connectivity
    new_reward = conct_init/cost_init
    conct_lst.append(iner_main_conct)
    conct_lst.append(connectivity)
    old_value = q_table[0][initiator_node]
    reward = reward_counting(landa, gama, new_reward, 0, old_value)
    print('init_reward: ', reward)
    value = value_count(old_value,0, reward, landa, gama)
    print('init_value: ', value)
    browse = []
    browse.append(initiator_node)
    s_lst = []
    s_lst.append(0)
    q_table[0][initiator_node] = value
    print('q_rable in initiating: ', q_table)
    #np.save('Q_table.npy', q_table)
    return q_table, s_lst , browse, value, reward, conct_lst, attack_list, iner_matrix, cost_init


def softmax(vector):
    e = exp(vector)
    v = e / e.sum()
    return v


def reward_counting(landa, gama, new_reward, next_value, old_value):

    #reward = landa*(new_reward + gama*next_value)
    reward = new_reward + gama*next_value
    return reward


def value_count(old_value, next_value,  reward, landa, gama):
    # ghablan lada dar reward zarb shode
    print('old_value: ', old_value )
    print ('next_value', next_value)
    print('reward: ' , reward)

    new_value = old_value +landa*(reward - old_value) #+ landa*(gama*(next_value)  - old_value)
    return new_value


def q_learning_convergence(learning_rate):
    continue_browsing= True
    max_delta= 0.0
    max_delta = max(learning_rate)
    print('max_delta:' , max_delta)
    if max_delta > 0.02:
        print('not converge')
        return continue_browsing
    else:
        continue_browsing = False
        print('continue_browsing is False: ', continue_browsing )
        return continue_browsing

def q_target_node(q_table,last_state, last_value, matrix,
                  active_lst, attack_list, conct, p,landa,gama,epsilon_prob):

    iner_matrix = deepcopy(matrix)
    internal_main_graph = create_main_graph(iner_matrix)
    active = deepcopy(active_lst)
    attack = deepcopy(attack_list)
    print('attack: ' , attack)
    cost = cost_count(internal_main_graph, active_lst, p)# be ezaye hameye node haye graph, cost hesab mishe
    reward = []
    numerate = [] # soorate kasre mohasebe reward
    for i in attack:
        temp_numer = []
        # br ezaye hame node haye active, disintegration anjam mishe va connectivity hesab mishe
        iner_matrix1 = deepcopy(iner_matrix)
        iner_attack, iner_matrix1 = disintegration(i, iner_matrix1, [])
        internal_main_graph = create_main_graph(iner_matrix1)
        connectivity = connectivity_count(internal_main_graph)
        subtrac = conct - connectivity # soorate kasre reward baraye i
        temp_numer.append(i)
        temp_numer.append(subtrac)
        numerate.append(temp_numer) #yek list az azaye dotayi sakhte mishe ke har ozv mige kodoom node ro age hamle konim soorate kasr chi mishe
    # hala ye cost darim ye list soorat baraye kasr ha
    # print('cost: ', cost)
    # print('nemerate: ', numerate)
    cost_for_attack = []
    for i in cost:
        if i[0] in attack:
            cost_for_attack.append(i)
    if len(cost_for_attack) != len(numerate):
        print('toolha yeki nist', "\n", 'len cost:',
              len(cost) ,'len numerate', len(numerate))
        return
    for i in range(len(numerate)): #mohasebeye reward baraye hameye azaye attack list
        r = []
        temp = numerate[i][1]/cost[i][1]
        r.append(numerate[i][0])
        r.append(temp) # in list dotayi hast. shomare node va reward
        reward.append(r)
    # print('reward list of attack list: ', reward)
    q_value = []
    for r in reward: # be ezaye hameye azaye active list q_value hesab mishavad
        temp_value = []
        value = q_table[last_state+1][r[0]]
        temp_value.append(r[0])
        temp_value.append(value)
        q_value.append(temp_value)
    if len(reward) != len(q_value):
        print('toolha yeki nist', "\n", 'len reward:',
              len(reward) ,'len numerate', len(q_value))
        return
    final_reward = []
    # print('q_value: ', q_value)
    # print('reward: ', reward)
    # print('len(q_value):', len(q_value),'len(reward): ',  len(reward))
    for i in range(len(reward)):
        temp_reward = []
        f_reward= reward_counting(landa, gama, reward[i][1],q_value[i][1], last_value)
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
        "node_number" : node,
        "reward": final_r,
        "softmax" : soft,
        })
    print('data_frame:' , data_frame)
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
        connectivity = connectivity_count(internal_main_graph2)
        subtrac2 = conct- connectivity # soorate kasre reward baraye i
        max_reward_epsilon = subtrac2/epsilon_cost
        epsilon_reward= reward_counting (landa, gama, 1,q_table[last_state+1][target_epsilon], last_value)
        return target_epsilon, 1


def q_learning(main_matrix, p, landa, gama, q_table, epsilon_prob, s_lst, browse , init_reward, conct_lst, attack_list, cost):
    sum_reward = 0
    sum_reward+= init_reward
    iner_matrix = deepcopy(main_matrix)
    active_nodes = active_node(iner_matrix)
    connectivity = conct_lst[-1]
    main_graph = create_main_graph(iner_matrix)
    delta = []
    while len(active_nodes) != 0:
        if len(active_nodes) == 0:
            print('delta:' , delta)
            print ('Network has disintegrated successfuly in Q_learning')
            continue_browsing = q_learning_convergence(delta)
            return conct_lst, cost, sum_reward, q_table, browse, continue_browsing
        else:
            #print('attack_list in else: ', attack_list)
            temp_attack = []
            for node in attack_list:
                #print('node:', node)
                if node in active_nodes:
                    temp_attack.append(node)
            attack_list = deepcopy(temp_attack)
        last_node = browse[-1]
        last_state = s_lst[-1]
        value = q_table[last_state][last_node]
        target_node, reward = q_target_node(q_table, last_state, value,
        iner_matrix, active_nodes, attack_list, connectivity, p, landa, gama, epsilon_prob)
        cost_internal = cost_count(main_graph, [target_node], p)
        #print('cost_internal : ', cost_internal)
        cost = cost + cost_internal[0][1]
        attack_list, iner_matrix_init = disintegration(target_node, iner_matrix, attack_list)
        main_graph = create_main_graph(iner_matrix)
        active_nodes = active_node(iner_matrix_init)
        sum_reward += reward
        browse.append(target_node)
        s_lst.append(s_lst[-1]+1)
        value = value_count(value, q_table[s_lst[-1]][browse[-1]], reward, landa, gama)
        print('value: ' , value )
        learning_rate =abs(q_table[s_lst[-2]][browse[-2]] - value)
        print('learning_rate : ' , learning_rate)
        delta.append(learning_rate)
        q_table[s_lst[-2]][browse[-2]] = value
        print('q_table:' , q_table)
        connectivity = connectivity_count(main_graph)
        conct_lst.append(connectivity)
        if len(active_nodes) == 0:
            print('delta:' , delta)
            print ('Network has been disintegrated successfuly in Q_learning')
            continue_browsing = q_learning_convergence(delta)
            return conct_lst, cost, sum_reward, q_table, browse , continue_browsing
    return conct_lst, cost, sum_reward, q_table, browse


def q_learning_base(p, landa, gama, epsilon_prob):
    q_table = np.load('Q_table.npy', allow_pickle=True)
    sum_value_nd = np.load('Sum_valu.npy' , allow_pickle= True)
    sum_value = sum_value_nd.tolist()
    with open('Initiator_node.pkl', 'rb') as handle:
        initiator_node = pickle.load(handle)
    print('initiator node: ', initiator_node)
    main_matrix = np.load('Main_Matrix.npy', allow_pickle= True)
    iner_main_matrix = deepcopy(main_matrix)
    continue_browsing = True
    i = 1480

    while continue_browsing:
        q_table, init_s_lst , init_browse, init_value, init_reward, init_conct_lst, init_attack_list, iner_matrix, cost_init = \
        q_initiator(main_matrix, p, landa, gama, q_table, initiator_node)
        conct_lst, cost, sum_reward, q_table, browse, continue_browsing = q_learning(main_matrix, p,landa, gama, q_table,
        epsilon_prob, init_s_lst, init_browse , init_reward, init_conct_lst, init_attack_list, cost_init)
        sum_value.append(sum_reward)
        print('browse:', browse)
        print('Q_Table', q_table)
        np.save('Q_table.npy', q_table)
        np.save('Sum_valu.npy', sum_value)
        i = i+1
        print('counter:', i)
        print('max valu of sum_value', max(sum_value))
        print('sum_value: ', sum_value)
        # if i > 40:
        #     continue_browsing = convergence_value(sum_value, max(sum_value))

    return q_table, i


#-----------------------Automata--------------------------------

def h_value_count_update( current_state, target_node, h_table, alfa):
    total_node = np.load('Total_Node.npy', allow_pickle= True)
    iner_total_node = deepcopy(total_node)
    Pi = h_table[current_state][target_node]
    h_table[current_state][target_node] = Pi + alfa*(1-Pi)
    i = current_state
    for j in range(iner_total_node):
        if j != target_node:
            h_table[i][j] == (1 - alfa) * h_table[i][j]
    print('h_table after update: ',h_table)
    return h_table


def automata_convergence(prob_lst):
    continue_browsing = True
    anthropy = 0
    for i in prob_lst:
        temp = (i * math.log(1/i))
        anthropy += temp
    if anthropy > 0.02:
        print('not converged')
        print('anthropy:'  , anthropy)
        return continue_browsing
    else:
        print('It is converged')
        print('anthropy:'  , anthropy)
        continue_browsing = False
        return continue_browsing


def h_initiator(main_matrix, p, alfa, h_table, initiator_node):
    main_conct = np.load('Main_Conct.npy')
    iner_main_conct = deepcopy(main_conct)
    cost = p
    iner_matrix = deepcopy(main_matrix)
    main_graph = create_main_graph(iner_matrix)
    active_lst = active_node(iner_matrix)
    cost_lst = cost_count(main_graph, active_lst, p)
    cost_init = cost_lst[initiator_node][1]
    attack_list, iner_matrix= disintegration(initiator_node, iner_matrix, [])
    main_graph = create_main_graph(iner_matrix)
    conct_lst = []
    connectivity = connectivity_count(main_graph)
    conct_init = iner_main_conct - connectivity
    new_reward = conct_init/cost_init
    conct_lst.append(iner_main_conct)
    conct_lst.append(connectivity)
    browse = []
    browse.append(initiator_node)
    s_lst = []
    s_lst.append(0)
    prob = []
    h_table = h_value_count_update( s_lst[-1], initiator_node, h_table, alfa)
    prob.append(h_table[s_lst[-1]][initiator_node])
    return h_table, s_lst, browse, conct_lst, attack_list, iner_matrix, cost_init, prob

def h_target_node(matrix, active_lst, attack_list, conct, p, epsilon_prob):
    iner_matrix = deepcopy(matrix)
    internal_main_graph = create_main_graph(iner_matrix)
    active = deepcopy(active_lst)
    attack = deepcopy(attack_list)
    print('attack: ' , attack)
    cost = cost_count(internal_main_graph, active_lst, p)# be ezaye hameye node haye graph, cost hesab mishe
    reward = []
    numerate = [] # soorate kasre mohasebe reward
    for i in attack:
        temp_numer = []
        # br ezaye hame node haye active, disintegration anjam mishe va connectivity hesab mishe
        iner_matrix1 = deepcopy(iner_matrix)
        iner_attack, iner_matrix1 = disintegration(i, iner_matrix1, [])
        internal_main_graph = create_main_graph(iner_matrix1)
        connectivity = connectivity_count(internal_main_graph)
        subtrac = conct - connectivity # soorate kasre reward baraye i
        temp_numer.append(i)
        temp_numer.append(subtrac)
        numerate.append(temp_numer) #yek list az azaye dotayi sakhte mishe ke har ozv mige kodoom node ro age hamle konim soorate kasr chi mishe
    cost_for_attack = []
    for i in cost:
        if i[0] in attack:
            cost_for_attack.append(i)
    if len(cost_for_attack) != len(numerate):
        print('toolha yeki nist', "\n", 'len cost:',
              len(cost) ,'len numerate', len(numerate))
        return
    for i in range(len(numerate)): #mohasebeye reward baraye hameye azaye attack list
        r = []
        temp = numerate[i][1]/cost[i][1]
        r.append(numerate[i][0])
        r.append(temp) # in list dotayi hast. shomare node va reward
        reward.append(r)
    # print('reward list of attack list: ', reward)
    node = []
    final_r = []
    for f in reward:
        node.append(f[0])
        final_r.append(f[1])
    soft = softmax(final_r)
    data_frame = pd.DataFrame({
        "node_number" : node,
        "reward": final_r,
        "softmax" : soft,
        })
    print('data_frame:' , data_frame)
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


def automata_learning(main_matrix, p, alfa, h_table, epsilon_prob, s_lst, browse , conct_lst, attack_list, cost, prob_lst):
    iner_matrix = deepcopy(main_matrix)
    active_nodes = active_node(iner_matrix)
    connectivity = conct_lst[-1]
    main_graph = create_main_graph(iner_matrix)
    prob = []
    while len(active_nodes) != 0:
        if len(active_nodes) == 0:

            print ('Network has disintegrated successfuly in automata_learning')
            continue_browsing = automata_convergence(prob_lst)
            return conct_lst, cost, h_table, browse, continue_browsing
        else:
            #print('attack_list in else: ', attack_list)
            temp_attack = []
            for node in attack_list:
                #print('node:', node)
                if node in active_nodes:
                    temp_attack.append(node)
            attack_list = deepcopy(temp_attack)
        target_node = h_target_node(iner_matrix, active_nodes, attack_list, connectivity, p, epsilon_prob)
        cost_internal = cost_count(main_graph, [target_node], p)
        #print('cost_internal : ', cost_internal)
        cost = cost + cost_internal[0][1]
        attack_list, iner_matrix_init = disintegration(target_node, iner_matrix, attack_list)
        main_graph = create_main_graph(iner_matrix)
        active_nodes = active_node(iner_matrix_init)
        browse.append(target_node)
        s_lst.append(s_lst[-1]+1)
        h_table = h_value_count_update(s_lst[-1], target_node, h_table, alfa)
        prob.append(h_table[s_lst[-1]][browse[-1]])
        print('h_table:' , h_table)
        connectivity = connectivity_count(main_graph)
        conct_lst.append(connectivity)
        if len(active_nodes) == 0:
            print ('Network has been disintegrated successfuly in automata_learning')
            continue_browsing = automata_convergence(prob_lst)
            return conct_lst, cost, h_table, browse, continue_browsing
    return conct_lst, cost, h_table, browse, continue_browsing


def h_learning_base(p, alfa, epsilon_prob):
    h_table = np.load('H_Table.npy', allow_pickle=True)
    with open('Initiator_node.pkl', 'rb') as handle:
        initiator_node = pickle.load(handle)
    print('initiator node: ', initiator_node)
    main_matrix = np.load('Main_Matrix.npy', allow_pickle= True)
    iner_main_matrix = deepcopy(main_matrix)
    continue_browsing = True
    i = 0
    while continue_browsing:
        h_table, init_s_lst , init_browse, init_conct_lst, init_attack_list, iner_matrix, cost_init, prob_lst= \
            h_initiator(main_matrix, p, alfa, h_table, initiator_node)
        conct_lst, cost, h_table, browse, continue_browsing = automata_learning(main_matrix, p, alfa, h_table,
        epsilon_prob, init_s_lst, init_browse, init_conct_lst, init_attack_list, cost_init, prob_lst)
        print('browse:', browse)
        print('Q_Table', h_table)
        np.save('H_Table.npy', h_table)
        i = i+1
        print('counter:', i)
        # if i > 40:
        #     continue_browsing = convergence_value(sum_value, max(sum_value))

    return h_table, i








#-------------Report____________
def plot_connect(con_rand, con_DC, con_BC, con_UW, con_Greedy, con_GA, con_DSQ ,con_DSA):

    list_name = []
    list_name.append('Random')
    list_name.append('DC')
    list_name.append('BC')
    list_name.append('UW')
    list_name.append('Greedy')
    list_name.append('GA')
    list_name.append('DSQ')
    list_name.append('DSA')
    episode = [len(con_rand), len(con_DC),len(con_BC), len(con_UW), len(con_Greedy), len(con_GA),len(con_DSQ) ,len(con_DSA)]
    min_episode = min(episode)
    order = []
    for i in range(min_episode):
        order.append(i)
    del con_rand[min_episode: len(con_rand)]
    del con_DC[min_episode: len(con_DC)]
    del con_BC[min_episode: len(con_BC)]
    del con_UW[min_episode: len(con_UW)]
    del con_Greedy[min_episode: len(con_Greedy)]
    del con_GA[min_episode: len(con_GA)]
    del con_DSQ[min_episode: len(con_DSQ)]
    del con_DSA[min_episode: len(con_DSA)]

    plt.plot(con_rand, label = 'Rand', lw=2, marker='s', ms=6) # square
    plt.plot(con_DC, label = 'DC', lw=2, marker='^', ms=6) # triangle
    plt.plot(con_BC, label = 'BC', lw=2, marker='o', ms=6) # circle
    plt.plot(con_UW, label = 'UW', lw=2, marker='D', ms=6) # diamond
    plt.plot(con_Greedy, label = 'Greedy', lw=2, marker='P', ms=6) # filled plus sign
    plt.plot(con_GA, label = 'GA', lw=2, marker='3', ms=6) # tri_left
    plt.plot(con_DSQ, label = 'DSQ', lw=2, marker='>', ms=6) # triangle_right
    plt.plot(con_DSA, label = 'DSA', lw=2, marker='+', ms=6) # plus
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


def table_view(cost_btw, cost_deg, cost_Rand, cost_weight, cost_GA, cost_greedy,cost_q, cost_aut):
    # generate matrix
    matrix = np.zeros((8, 5), dtype="float", order='c')

    matrix[0] = cost_btw
    matrix[1] = cost_deg
    matrix[2] = cost_Rand
    matrix[3] = cost_weight
    matrix[4] = cost_GA
    matrix[5] = cost_greedy
    matrix[6] = cost_q
    matrix[7] = cost_aut
    print(matrix.dtype)
    print(matrix)
    #plot the matrix as an image with an appropriate colormap
    #matrix_in = np.random.uniform(0,1,(8,5))
    # for j in range(5):
    #     matrix_in[0][j] = cost_deg[j]
    # for j in range(5):
    #     matrix_in[1][j] = cost_deg[j]
    # for j in range(5):
    #     matrix_in[2][j] = cost_Rand[j]
    # for j in range(5):
    #     matrix_in[3][j] = cost_weight[j]
    # for j in range(5):
    #     matrix_in[4][j] = cost_GA[j]
    # for j in range(5):
    #     matrix_in[5][j] = cost_greedy[j]
    # for j in range(5):
    #     matrix_in[6][j] = cost_q[j]
    # for j in range(5):
    #     matrix_in[7][j] = cost_aut[j]

    print(matrix[1][2])
    print(matrix)
    print(matrix.dtype)
    plt.imshow(matrix.T, aspect='auto', cmap="bwr")

    # add the values
    for (i, j), value in np.ndenumerate(matrix):

        plt.text(i, j, "%.3f"%value, va='center', ha='center')
    plt.axis('on')
    plt.show()
    #plt.imshow()
    return




#-------------MAIN------------------------------------------------------------------

#-------------initiator--------------
# #
# list_node_initial , Layen_Count = list_node_init()
# # np.save('list_node_initial' , list_node , allow_pickle=True)
# # np.save('Layen_Count' , layer_n , allow_pickle=True)
# print('1')
# Total_Matrix = create_matrix(list_node_initial)
# # #np.save('Total_Matrix' , total_mtrx , allow_pickle=True)
# # print('2')
# List_Struct= list_struc(list_node_initial)
# # #np.save('List_Struct' , List_Struct , allow_pickle=True)
# # print('3')
# comb_dis = create_comb_array(list_node_initial)
# # #np.save('comb_dis' , List_Struct , allow_pickle=True)
# print('4')
# list_of_nodes , Label = Create_List_of_Nodes(List_Struct)
# # # np.save('list_of_nodes' , list_of_nodes , allow_pickle=True)
# # # np.save('Label' , Label , allow_pickle=True)
# print('5')
# Map_dic, Total_Node = node_Mapping(list_of_nodes)
# # # np.save('Map_dic' , map_dic , allow_pickle=True)
# # # np.save('Total_Node' , i , allow_pickle=True)
# print('6')
# Attack_Nodes = random_atthck_nodes(list_of_nodes)
# # #np.save('Attack_Nodes' , map_dic , allow_pickle=True)
# print('7')
# Attack_Map = attack_maping(Attack_Nodes, Map_dic)
# # # np.save('Attack_Map' , map_dic , allow_pickle=True)
# print('8')
# Main_Matrix = create_major_matrix(Total_Matrix , Layen_Count)
# # #np.save('Main_Matrix' , Main_Matrix , allow_pickle=True)
# print('9')
# Main_Graph = create_main_graph_init(Main_Matrix)
# # # np.save('Main_Graph' , Main_Graph , allow_pickle=True)
# print('10')
# Main_Conct = connectivity_count_init(Main_Graph)
# # # np.save('Main_Conct' , Main_Conct , allow_pickle=True)
# print('11')
# Triple_Weight  = weight_def (Main_Matrix)
# # # np.save('Triple_Weight' , Triple_Weight ,  allow_pickle=True)
# print('12')
# Active_Node = active_node_init(Main_Matrix)
# # print('Active_node', Active_Node)
# # #np.save('Active_Node' , Active_Node ,  allow_pickle=True)
# print('13')
# Averg_Weight = weight_account_init(Triple_Weight, Active_Node)
# # print('Averg_Weight' , Averg_Weight)
# # # np.save('Averg_Weight' , Averg_Weight , allow_pickle=True)
# print('14')
# Q_Table = table_initiator_Q(Total_Node)
# print('15')
# H_Table = h_table_initiator(Total_Node)
# print('16')
# Initiator_Node = rand_node()
# print('17')
# print('initializing has finished successfully')


#-----------------methodes-------------------

# Rand_Node = rand_node()
# Connectivity_BTW, Cost_BTW = closeness_dis(1)
# print('cost_btw:' , Cost_BTW)
# Connectivity_DEG, Cost_DEG = closeness_dis(2)
# print('cost_DEG:' , Cost_DEG)
# Connectivity_Random , Cost_Rand = random_recursive_dis()
# print('cost_Rand:' , Cost_Rand)
# Connectivity_Weight, Cost_Weight = weight_recursive_dis()
# print('cost_weight:' , Cost_Weight)
# Connectivity_Greedy, Cost_Greedy = Greedy_disintegration()
# print('cost_greedy:', Cost_Greedy)
# Connectivity_GA, Cost_GA = GA_dis( 0.9, 0.05, 30)
# print('cost_GA:' , Cost_GA)

# Connctivity_Q, Cost_Q = Q_cost_creation( [0.0, 0.5, 1.0, 1.5, 2.0], 0.1, 0.9)
# print('cost_Q' , Cost_Q)
# Connctivity_Q, Cost_q , Q_value, Target_Node_Lst_Q = q_learning(Main_Matrix , 0.0 , 0.1 , 0.9)
# print('Connctivity_q:' , Connctivity_Q,'Cost_q:',  Cost_q ,'Q_value:',  Q_value)

# Q_Table, i = q_learning_base(1, 0.1, 0.5, 0.1)
h_learning_base(1, 0.05, 0.1)




# H_Table , i  = automata_learn_convergence( 1, 0.3)
# conct_lst, cost, Target_Node_Lst_AUT = automata_dis(Rand_Node, 0.0)
# Connctivity_aut, Cost_aut = automata_cost_creation(  [0.0, 0.5, 1.0, 1.5, 2.0])
# print('cost_aut' , Cost_aut)
# print('Target_Node_Lst_Q: ' , Target_Node_Lst_Q)
# print ('Target_Node_Lst_AUT: ', Target_Node_Lst_AUT)










#--------------------Reports-----------------------
#plot_connect(Connectivity_Random, Connectivity_DEG, Connectivity_BTW, Connectivity_Weight, Connectivity_Greedy, Connectivity_GA, Connctivity_Q ,Connctivity_aut)
#table_view(Cost_BTW, Cost_DEG, Cost_Rand, Cost_Weight, Cost_GA, Cost_Greedy, Cost_Q, Cost_aut)
#plot_connect(con_rand, con_DC, con_BC, con_UW, con_Greedy,con_GA , con_Q ,con_DSA)



# total_sum = np.load('Sum_valu_list.npy' , allow_pickle= True)
# print('sum_reward' , total_sum)
#
# q_table  = np.load('Q_table.npy' , allow_pickle= True)
# print('q_table: ', q_table)




