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
from copy import copy , deepcopy
from sklearn.preprocessing import normalize
import pandas as pd
from operator import itemgetter
import matplotlib.pyplot as plt
import ga



def list_node():
    """gets layers count as an int variation layer_n
    for each layer creates a random int number of nodes as a member of list_node
    returns list of layer nodes as list_node"""
    layer_n = int(input("input a number as number of layers: "))
    temp = itertools.count(1)
    index = [next(temp) for i in range(layer_n)]
    # print('index:', index)
    list_node = []
    for i in index:
        node = np.random.randint(30, 40)
        list_node.append(node)
    #print('list_node:', list_node)
    return list_node , layer_n


def list_struc(list_node):
    # gets the number of layers, creates and returns struct
    struc = []
    for x in range(len(list_node)):
        str = [x, list_node[x]]
        struc.append(str)

    #print('struc:', struc)
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
    weights = [np.random.randint(10, 20) for r in range(m)]
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
    return comb_array #for each member of combarray: 0.comb, 1.B0, 2.B1, 3.B2, 4.B3, 5.R(edge_number), 6.n(lenth)


def create_matrix(list_node):
    """ from list_node() gets list_node as each layer nodes number
    create a zero matrix
    for each layer create a random weighted graph by using of node number
    of layer: random_weighted_graph(nlist[l-1])
    create adjacency matrix for each matrix using g.get_adjacency() """
    n = len(list_node)  # number of layers , list_node= list of nodes in each layer
    #print ('layer number:', n)
    total_mtrx = np.zeros((n, n), dtype="object", order='c')
    # total_mtrx = np.empty((n, n, n, n ), dtype="int", order='c')
    #print('total_mtrx:', total_mtrx)
    i = 0
    j = 0
    edge_list = []
    """create diagonal of main matrix """

    for l in range(n): # n = number of layers
        #print ('l:', l)
        m = list_node[l]
        #print ('l:' , l , 'list_node[l]: ' , m)
        adjc_mtrx = random_weighted_graph(m) # craete random weighted graph for each layer. l = number of nodes
        #adjc_mtrx = g.get_adjacency()
        #print('xxxxxx:', adjc_mtrx)
        total_mtrx[i][j] = adjc_mtrx
        i = i + 1
        j = j + 1

    #create Bipartite matrixes
    comb = create_comb_array(list_node) #for each member of combarray: 0.comb, 1.B0, 2.B1, 3.B2, 4.B3, 5.R(edge_number), 6.n(lenth)
    for p in comb:
        #print ('number of biprt edge:' , p[5], 'B1:', p[2] , 'B3', p[4])
        G = nx.bipartite.gnmk_random_graph(p[2], p[4], p[5])
        G_lenth =(p[6])
        Row_order = range(G_lenth)
        #print ('row_order:', Row_order)
        #G_adj0 = np.zeros((G_lenth, G_lenth), order='c')
        G_adj = nx.bipartite.biadjacency_matrix(G,row_order=Row_order, column_order= Row_order)
        # print('Type of G_adj0:',type(G_adj))
        # print('G_adj:', G_adj)
        # print ('shape:' ,csr_matrix.get_shape(G_adj))
        #G_dens = csr_matrix.todense(G_adj)
        G_array = csr_matrix.toarray(G_adj)
        #print('G_array:', G_array, G_array.ndim , G_array[0], len(G_array[0]), G_array[0][0])

        dens_matrx= np.zeros((G_lenth, G_lenth), order='c')
        for i in range(G_lenth):
            for j in range(G_lenth):
                dens_matrx[i][j]= int(G_array[i][j])
        #print('dens_matrx:', dens_matrx)

        for i in range(0 , G_lenth):
            for j in range(0, G_lenth):
                if dens_matrx[i][j]==1:
                    dens_matrx[j][i]= 1

        #print('dens_matrx after convertion :', dens_matrx)
        total_mtrx[p[1]][p[3]] = dens_matrx
        total_mtrx[p[3]][p[1]] = dens_matrx


    # print('tot:', total_mtrx)
    # print(len(total_mtrx))
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


    return list_of_Node , list_of_Node_lables


def random_atthck_nodes(list_of_nodes):
    #print('len(list_of_nodes):',len(list_of_nodes))

    attacked_number = math.floor(len(list_of_nodes)/4)
    #print('attacken number on nodes:', attacked_number)
    attacked_list = random.sample(list_of_nodes, attacked_number)
    #print('attacked nodes:', attacked_list)
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
    node_struct = []
    for node in attacked_nodes:
        node_struct_temp = []
        layer = node[0]
        node_num = node[1]# index of attacked node in its layer
        Struct = List_Struct[layer]# get struct if the layer that belongs to attacked node
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


def node_Mapping (list_of_Node):
    i = 0
    map_dic = {}
    for node in list_of_Node:
        map_dic[i] = node
        i = i+1
    return map_dic , i


def create_index_list(i , total_node):
    index_list = []
    for a in range(total_node):
        temp_node = Map_dic[a]
        if temp_node[0] == i:
            index_list.append(a)
    return index_list


def create_major_matrix(Total_Matrix, Layer_Count):
    # main graph ro misaze
    # Map_dic, Total_Node = node_Mapping(list_of_nodes) / Inha ro darim
    # list_node_initial , Layen_Count = list_node()
    main_matrix = np.zeros((Total_Node, Total_Node), dtype="int", order='c')
    index_list = []
    temp_node = []
    z = 0
    for i in range(Layer_Count):
        for j in range(Layer_Count):
            matrix = Total_Matrix[i][j]
            if i == j:
                index_list = create_index_list(i, Total_Node)
                for b in range(len(index_list)):
                    for c in range(len(index_list)):
                        main_matrix[index_list[b]][index_list[c]] = matrix[b][c]


            else:
                bi_index_list = []
                index_list1 = create_index_list(i , Total_Node)
                index_list2 = create_index_list(j , Total_Node)
                for mp in index_list2:
                    index_list1.append(mp)
                    bi_index_list = index_list1

                for bi in range(len(bi_index_list)):
                    for ci in range(len(bi_index_list)):
                            if matrix[bi][ci]==1:
                                main_matrix[bi_index_list[bi]][bi_index_list[ci]] = matrix[bi][ci]


    main = np.matrix(np.array(main_matrix))

    return main_matrix


def create_main_graph(adjacency_matrix, labels):
    # main graph ro namayesh midim
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    #nx.draw(gr, node_size=500,  with_labels=True)
    #plt.show()
    return gr


def create_main_graph_copy(adjacency_matrix, labels):
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


def attack_Node_Mapping (attack_list):
    # node haye attack ro be shomare haye jadid map mikone
    index_list = []
    for a in range(Total_Node):
        temp_node = Map_dic[a]
        for node in attack_list:
            if temp_node == node:
                index_list.append(a)
    return index_list


def connectivity_count(main_graph):
    connectivity = nx.average_node_connectivity(main_graph)
    return connectivity


def disintegration (node, main_matrix, attack_list):
    # node: the first index of attack node list
    # main_matrix: matrix should updated by each disintegration step
    # attack_List: should update by each disintegration step
    # type = 1: Random / type = 2: DEG / Type = 3: BWN / Type = 4: WGHT

    neigh = []
    #print('in disintegrate:')
    #print (main_matrix , "\n", 'Node for attack: ', node)
    #print('Total_Node:', Total_Node, "\n", 'attack_list:',attack_list)
    for i in range(Total_Node):
        #print('i:', i, 'node', node)
        if main_matrix[i][node] == 1:
            neigh.append(i)
            main_matrix[i][node] = 0
            main_matrix[node][i] = 0
    #index = attack_list.index(node)
    #attack_list.pop(index)

    for n in neigh:
        if n not in attack_list:
            attack_list.append(n)
    final_attack_list = list(set(attack_list))
    print('diiiiiiiiiiiiiiissssssssssssssssssssss')
    return final_attack_list , main_matrix


def closeness_dis(type , main_matrix):
    # aval bayad ye peygham neshoon bedim ke in che noe disi hast
    cost_lst = [0.0, 0.0, 0.0, 0.0, 0.0]
    p = [0.0, 0.5, 1, 1.5, 2]
    iner_matrix = deepcopy(main_matrix)
    main_graph = create_main_graph(iner_matrix, Label)
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
            print('final main matrix for other methodes: ', Main_Matrix)
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
            main_graph = create_main_graph(iner_matrix, Label)
            connectivity = connectivity_count(main_graph)
            conct = (connectivity/Main_Conct)
            connectivity_lst.append(conct)

            print('connectivity_lst', connectivity_lst)


def random_recursive_dis(main_matrix):
     cost_lst = [0.0, 0.0, 0.0, 0.0, 0.0]
     p = [0.0, 0.5, 1.0, 1.5, 2]
     iner_matrix = deepcopy(main_matrix)
     main_graph = create_main_graph(iner_matrix, Label)
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
            print('final main matrix for other methodes: ', Main_Matrix)
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
            main_graph = create_main_graph(iner_matrix, Label)
            connectivity = connectivity_count(main_graph)
            conct = (connectivity/Main_Conct)
            connectivity_lst.append(conct)


def weight_def (main_matrix):
    # be ezaye har node ye vazne tasadofi ijad mikone va liste node haye faal ro ham tashkhis mide va barmigardoone
    list_of_weight = []

    for i in range(Total_Node):
        for j in range(Total_Node):
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
    return list_of_weight


def active_node(main_matrix):
    #har bar ke matrix ro update mikonm va azash kam mishe in metode mire node haye zendash ro list mikone
    active_node =  []
    for i in range(Total_Node):
        for j in range(Total_Node):
            node1 = []
            if main_matrix[i][j] ==1:
                node1.append(i)
                node1.append(j)
                active_node.append(node1)
    active_node_list = []
    for node in active_node:
        if node[0] not in active_node_list:
            active_node_list.append(node[0])
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


def weight_account(list_of_weight , active_nodes ):
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
        #= sorted(node_and_avr_list)
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


def weight_recursive_dis(main_matrix):
     # recursive disintegration ro anjam mide
     #iner_main_matrix = [row[:] for row in main_matrix]
     cost_lst = [0.0, 0.0, 0.0, 0.0, 0.0]
     p = [0.0, 0.5, 1.0, 1.5, 2]
     iner_main_matrix = deepcopy(main_matrix)
     main_graph = create_main_graph(iner_main_matrix, Label)
     connectivity_lst = []
     connectivity_lst.append(1)
     attack_list = []
     list_of_weight  = weight_def (iner_main_matrix)
     primitive_list_of_weight = deepcopy(list_of_weight)#******************
     #np.save('primitive_list_of_weight' , primitive_list_of_weight)
     active_nodes = active_node(iner_main_matrix)
     node_averg = weight_account(list_of_weight, active_nodes)
     primitive_node_avrg = deepcopy(node_averg)#**********************
     #np.save('primitive_list_of_weight' , primitive_list_of_weight)
     attack_list.append(node_averg[0][1])
     # print ('attack_list:;;;;;;;;;;;;;;;;', attack_list)
     # print('intiatig;;;;;;;;;;;;;;')
     while len(active_nodes) != 0:
        #print('111111111111111')
        for node in attack_list:
            #print('222222222222222')
            if node not in active_nodes:
                index = attack_list.index(node)
                attack_list.pop(index)
                print('alone node hase deleted: ', node)
        if len(active_nodes)!= 0 and len(attack_list)==0:
                #print('333333333333333')
                # baraye jologiri az khata ya loop binahayat da moredi ke graph az aval chand bakhshi boode
                main_graph = create_main_graph(iner_main_matrix, Label)
                list_of_weight  = weight_def (iner_main_matrix)
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
                main_graph = create_main_graph(iner_main_matrix, Label)
                connectivity = connectivity_count(main_graph)
                conct = (connectivity/Main_Conct)
                connectivity_lst.append(conct)
                #print('attack_list', attack_list)
                #list_of_weight  = weight_def (iner_main_matrix)
                active_nodes = active_node(iner_main_matrix)
                node_averg = weight_account(list_of_weight, active_nodes)
                attack_list = attack_weight_sort(attack_list , node_averg)
     if len(active_nodes) == 0:
            print ('Network has disintegrated successfuly by wight method ')
            return primitive_node_avrg , primitive_list_of_weight , connectivity_lst , cost_lst


def attack_maping(attack_list, map_dic):
    attack_map = []
    for node in attack_list:
        for n in map_dic:
            if node == map_dic[n]:
                attack_map.append(n)
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

def fitness_count(primitive_weight_duble, primitive_weight_triple, main_graph , initiator ):
    weight_list_avrg = deepcopy(primitive_weight_duble)
    weight_list_triple = deepcopy(primitive_weight_triple)
    bc = closeness_btw(main_graph)
    dc = closeness_deg(main_graph)
    bc_sort = sorted(bc.items(), key=lambda x: x[1], reverse=True)
    dc_sort = sorted(dc.items(), key=lambda x: x[1], reverse=True)
    #weight_list_avrg = weight_account_copy(weight_list_triple , attack_list)
    print('weight: ', weight_list_avrg, "\n", 'weight_list_triple: ',weight_list_triple ,  "\n", 'bc: ', bc_sort ,"\n",  'dc', dc_sort)
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
    print('len initiator:' , len(initiator))
    print('methode: fitness_count: ', 'len weight_lst:', len(weight_lst) , 'len dc_lst:' , len(dc_lst)  ,'len bc_lst: ', len( bc_lst))
    return weight_lst , dc_lst, bc_lst


def fitness_arrenge(bc, dc, uw, initiator):
    print('bc:', bc, "\n", 'dc:', dc, "\n", 'uw:', uw)
    create_dataset = False
    for i in range(len(bc)):
        if bc[i][0] == dc[i][0] and bc[i][0] == uw[i][0]:
            create_dataset = True
        else:
            print('index ha ba ham yeki nistan', bc[i][0], dc[i][0],  uw[i][0])
            for node in uw:
                if node[i][0] == uw[i][0]:
                    index = uw.index(node)
                    uw.pop(index)
    if len(bc)== len(dc) and len(dc) == len(uw):
        print('toolha ba ham barabaran')
    else:
        print('toolha ba ham yeki nist', len(bc), len(dc) , len(uw))
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

    # select cross
    children = random.sample(iner_init_cross_lst, cross_count)
    parent = []
    for i in iner_init_cross_lst:
        if i not in children:
            parent.append(i)
    parent_final = random.sample(parent , parent_count)

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
    neigh = []
    neigh_single = []
    neigh_final = []
    for i in children:
        for j in range(Total_Node):
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
    main_graph = create_main_graph(iner_matrix, Label)
    active_nodes = active_node(iner_matrix)
    initiator = random_atthck_nodes(active_nodes)
    generation_size = len (initiator)
    print('methode: list_initiate: ','len initiate list: ', len(initiator), "\n",  'initiate list:' , initiator)
    return initiator, generation_size


def list_sorting(primitive_weight_duble, primitive_weight_triple, main_graph , initiate_lst):
    weight_lst , dc_lst, bc_lst = fitness_count(primitive_weight_duble, primitive_weight_triple, main_graph , initiate_lst )
    #bc , dc, uw = list_allignment(bc_lst, dc_lst, weight_lst)
    sorted_lst = fitness_arrenge( weight_lst , dc_lst, bc_lst, initiate_lst)
    return sorted_lst


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
               primitive_weight_duble, primitive_weight_triple ):
    iner_init = deepcopy(initiate_lst)
    iner_matrix = deepcopy(main_matrix)
    print('sorted_lst in GA_target_node:', initiate_lst)
    print('sorted_lst type: ', type(initiate_lst))
    print('evolution: ', evolution)
    print('generation_size:' , generation_size)
    i = 0
    while i < evolution:
        if len(iner_init) < generation_size:
            print('list is shorter than generation')
            target_node = iner_init[-1]
            return target_node

        last_gen = list_sorting(primitive_weight_duble, primitive_weight_triple, main_graph , iner_init)
        print('last_gen in first step:' , last_gen)
        target_node = last_gen[-1]

        print('target_node on first step:', target_node)
        last_gen_len = len(last_gen)
        mut , permanent_parent_lst , children = split_initial_lst(last_gen, generation_size,  mutation_portion, crossover_portion)
        children_len = len(children)
        crossover_lst = crossover(children, iner_matrix, last_gen)
        print('crossover before split in target_node methode: ', crossover_lst)
        crossover_lst_sorted = list_sorting(primitive_weight_duble, primitive_weight_triple, main_graph , crossover_lst)
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


def GA_dis (main_matrix , primitive_weight_duble, primitive_weight_triple , crossover, mutation_portion, evolution):
    cost_lst = [0.0, 0.0, 0.0, 0.0, 0.0]
    p = [0.0, 0.5, 1.0, 1.5, 2]
    #initiator = deepcopy(primitive)
    #init_len = len(initiator)
    iner_matrix = deepcopy(main_matrix)
    main_graph = create_main_graph(iner_matrix, Label)
    connectivity_lst = []
    connectivity_lst.append(1)
    active_nodes = active_node(iner_matrix)
    while len(active_nodes) != 0:

        if len(active_nodes) == 0:
            print ('Network has disintegrated successfuly in GA')
            return connectivity_lst, cost_lst

        initiate_lst , generation_size = list_initiate(iner_matrix)
        for node in initiate_lst:
            if node not in active_nodes:
                index = initiate_lst.index(node)
                initiate_lst.pop(index)
                print('alone node hase deleted: ', node)

        target_node = GA_target_node(mutation_portion , crossover, initiate_lst, generation_size , main_graph, main_matrix, evolution,
               primitive_weight_duble, primitive_weight_triple )
        print('target_node in GA_dis:', target_node)
        for i in range(len(p)):
            cost = cost_count(main_graph, [target_node], p[i])
            cost_lst[i] = cost_lst[i] + cost[0][1]
        attack_lst, iner_matrix = disintegration(target_node, iner_matrix, [])
        print('primitive_weight_triple:' , primitive_weight_triple)
        active_nodes = active_node(iner_matrix)
        print('active_node after dis:', active_nodes)
        main_graph = create_main_graph(iner_matrix, Label)



        connectivity = connectivity_count(main_graph)
        conct = (connectivity/Main_Conct)
        connectivity_lst.append(conct)
        if len(active_nodes)== 0:
            print ('Network has disintegrated successfuly in GA')
            return connectivity_lst, cost_lst
    return connectivity_lst, cost_lst


#___________________Greedy_____________
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

def Greedy_disintegration (main_matrix, map_dic , primitive_averg_weight_duble, primitive_weight_triple):
    cost_lst = [0.0, 0.0, 0.0, 0.0, 0.0]
    p = [0.0, 0.5, 1.0, 1.5, 2]
    iner_matrix = deepcopy(main_matrix)
    main_graph = create_main_graph(iner_matrix, Label)
    connectivity_lst = []
    connectivity_lst.append(1)
    print('connectivity initiationg: ' , connectivity_lst)
    weight_list_avrg = deepcopy(primitive_averg_weight_duble)
    weight_list_triple = deepcopy(primitive_weight_triple)
    bc = closeness_btw(main_graph)
    dc = closeness_deg(main_graph)
    bc_sort = sorted(bc.items(), key=lambda x: x[1], reverse=True)
    dc_sort = sorted(dc.items(), key=lambda x: x[1], reverse=True)
    # print('len(primitive_averg_weight_duble)', len(primitive_averg_weight_duble))
    # print(' len(primitive_weight_triple)',  len(primitive_weight_triple))
    # print('len(map_dic)',len(map_dic))
    # print('bc_sort:', len(bc_sort))
    # print('dc_sort' , len(dc_sort))
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
    active_nodes = active_node(iner_matrix)
    bc_normal = sorted(bc_normal, key=itemgetter(0))
    dc_normal = sorted(dc_normal, key=itemgetter(0))
    target_node = target_choose(bc_normal, dc_normal, weight_normal)
    print('target_node:', target_node)
    while len(active_nodes) != 0:
         print('active_nodes in do while:',active_nodes)
         if len(active_nodes) == 0:
             print('Network has disintegrated successfuly in Greedy')
             print('connectivity in return: ', connectivity_lst)
             return connectivity_lst, cost_lst
         else:
             # if len(active_nodes)!=0 and len (attack_lst) ==0 :
             #        weight_list_avrg = deepcopy(primitive_averg_weight_duble)
             #        weight_list_triple = deepcopy(primitive_weight_triple)
             #        bc = closeness_btw(main_graph)
             #        dc = closeness_deg(main_graph)
             #        bc_sort = sorted(bc.items(), key=lambda x: x[1], reverse=True)
             #        dc_sort = sorted(dc.items(), key=lambda x: x[1], reverse=True)
             #        weight_list_reverse = []
             #        for n in weight_list_avrg:
             #            temp_n = []
             #            temp_n.append(n[1])
             #            temp_n.append(n[0])
             #            weight_list_reverse.append(temp_n)
             #        weight_list_reverse_sort = sorted(weight_list_reverse, key=itemgetter(0))
             #        weight_normal = normalize(weight_list_reverse_sort)
             #        bc_normal = normalize(bc_sort)
             #        dc_normal = normalize(dc_sort)
             #        active_nodes = active_node(iner_matrix)
             #        bc_normal = sorted(bc_normal, key=itemgetter(0))
             #        dc_normal = sorted(dc_normal, key=itemgetter(0))
             #        target_node = parent_choose(bc_normal, dc_normal, weight_normal)
             for i in range(len(p)):
                cost = cost_count(main_graph, [target_node], p[i])
                cost_lst[i] = cost_lst[i] + cost[0][1]
             attack_lst, iner_matrix = disintegration(target_node, iner_matrix, [])
             print('attack_lst after disintegration :', attack_lst)
             active_nodes = active_node(iner_matrix)
             if len(active_nodes)== 0:
                 print ('Network has disintegrated successfuly in Greedy')
                 return connectivity_lst, cost_lst
             main_graph = create_main_graph(iner_matrix, Label)
             connectivity = connectivity_count(main_graph)
             conct = (connectivity/Main_Conct)
             connectivity_lst.append(conct)
             print('connectivity in while: ', connectivity_lst)
             bc = closeness_btw(main_graph)
             dc = closeness_deg(main_graph)
             bc_sort = sorted(bc.items(), key=lambda x: x[1], reverse=True)
             dc_sort = sorted(dc.items(), key=lambda x: x[1], reverse=True)
             #dobare vazn nodha ro hesab mikonim
             print('weight_list_reverse:  in recursive: ' , weight_list_reverse , "\n", 'active_nodes in recursive:' , active_nodes)
             node_averg = weight_account_copy(weight_list_triple, active_nodes)
             recursive_wright = attack_weight_sort_copy(active_nodes , node_averg )
             weight_list_reverse = []
             for n in node_averg:
                 temp_n = []
                 temp_n.append(n[1])
                 temp_n.append(n[0])
                 weight_list_reverse.append(temp_n)
             weight_normal = normalize(weight_list_reverse)
             bc_normal = normalize(bc_sort)
             dc_normal = normalize(dc_sort)
             weight_list_reverse_sort = sorted(weight_list_reverse, key=itemgetter(0))
             weight_normal = normalize(weight_list_reverse_sort)
             bc_normal = sorted(bc_normal, key=itemgetter(0))
             dc_normal = sorted(dc_normal, key=itemgetter(0))
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


def table_initiator_Q(total_node):
    # Create Q-Table by Total_node Dimentions
    #n = len(total_node)
    q_table = np.zeros((total_node, total_node), dtype="float", order='c')
    print('q_table' , q_table)
    np.save('Q_table.npy' , q_table)
    return q_table


def cost_count(main_graph , actv_lst , p):
    print('active list in cost_count:', actv_lst)
    # just cost counting for each member of attack list
    #p = [0, 0.5, 1, 1.5, 2]
    degree = closeness_deg(main_graph)
    degree_lst = sorted(degree.items(), key=lambda x: x[1], reverse=True)
    sum_p = 0.0
    for i in degree_lst: # makhraje kasr ro inja misazim va rooye hameye grapg hast
        sum_p = sum_p +(i[1]**p)
    cost = []
    for i in actv_lst: # soorate kasr faghat baraye azaye attack_lst
        internal_cost = []
        for j in degree_lst:
            if i == j[0]:
                cost_p = j[1]**p
                c = (cost_p/sum_p)*25 # mohasebeye kasr be ezaye p haye mokhtalef
                internal_cost.append(i) # sakhte yek list ke har ozv an yek  node az attack_lst ast va hazine ba p haye mokhtalef
                internal_cost.append(c)
                cost.append(internal_cost)


    print('cost::::' , cost)

    return cost


def rand_node(main_graph):
    closeness = closeness_deg(main_graph)
    sort_order = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
    node = sort_order[np.random.randint(0, len(sort_order))][0]
    return node


#__________Automata_lerning______________


def target_node_aut_learning( matrix , active_lst , conct , p):

    iner_matrix = deepcopy(matrix)
    internal_main_graph = create_main_graph(iner_matrix, Label)
    active = deepcopy(active_lst)
    print('len(active_lst):' , len(active_lst), "\n", 'active_list:', active_lst)
    cost = cost_count(internal_main_graph, active_lst , p) #
    reward = []
    numerate = [] # soorate kasre mohasebe reward
    for i in active_lst:
        paire = []
        iner_matrix1 = deepcopy(iner_matrix)
        list, iner_matrix1 = disintegration(i, iner_matrix1, active)
        print('00000000000000000000000')
        internal_main_graph = create_main_graph_copy(iner_matrix1, Label)
        connectivity = connectivity_count(internal_main_graph)
        inter_con = (connectivity /Main_Conct)
        subtrac = conct- inter_con
        paire.append(i)
        paire.append(subtrac)
        numerate.append(paire) #yek list az azaye dotayi sakhte mishe ke har ozv mige kodoom node ro age hamle konim soorate kasr chi mishe
        print('i:' , i , 'paire:', paire)
        print('soorate kasre reward:', numerate)
        iner_matrix = deepcopy(matrix)
    # hala ye cost darim ye list soorat baraye kasr ha
    print('cost: ', cost)
    print('active_list: ', active_lst)
    print('attack:', active)
    print('list:', list)
    if len(cost) != len(numerate):
        print('lenths are not equal', "\n", 'len cost:', len(cost) ,'len numerate', len(numerate) )
        return
    for i in range(len(cost)):
        if cost[i][0] != numerate[i][0]:
            print('odrers are not sync')
            return
        else:
            r = []
            temp = numerate[i][1]/cost[i][1]
            r.append(cost[i][0])
            r.append(temp)
            reward.append(r) # in list dotayi hast. shomare node va reward
    print(reward)
    node = []
    reward_pure = []
    for i in reward:
        node.append(i[0])
        reward_pure.append(i[1])
    data_frame = pd.DataFrame({
        "node_number" : node,
        "q_value" : reward_pure,
        })
    target_decision = []
    print(data_frame)
    column0 = data_frame["q_value"]
    max_reward = column0.max()
    target_node_p = data_frame['node_number'][data_frame[data_frame['q_value'] == max_reward].index.tolist()].tolist()
    target_decision.append(target_node_p[0])
    print('target_decision:' , target_decision)
    # del(iner_matrix)
    # del(matrix)
    return target_decision[0] , max_reward



def h_value_count_update( current_state, target_node , h_table , a):

    Pi = h_table[current_state][target_node]
    h_table[current_state][target_node] = Pi + a*(1-Pi)
    i = current_state
    for j in range(Total_Node):
        if j != target_node:
            h_table[i][j] == (1 - a)* h_table[i][j]
    print('h_table after update: ' ,h_table)
    return h_table



def automata_dis(main_matrix, p , a, h_table ):
    cost = 0.0
    iner_matrix = deepcopy(main_matrix)
    main_graph = create_main_graph_copy(iner_matrix, Label)
    attack_list = []
    initiator_node = rand_node(main_graph)
    #h_table = table_initiator(Total_Node)
    attack_list, iner_matrix = disintegration(initiator_node, iner_matrix, attack_list)
    print('attack lst after automata init: ' , attack_list)
    main_graph = create_main_graph_copy(iner_matrix, Label)
    closeness = closeness_deg(main_graph)
    conct_lst = []
    conct_lst.append(1)
    connectivity = connectivity_count(main_graph)
    conct = (connectivity/Main_Conct)
    print('first connectivity: ', conct)
    conct_lst.append(conct)
    browse = []
    browse.append(initiator_node)
    #list of browsed nodes for disintegrating
    target_nodes_lst = []
    first_node = []
    first_node.append(initiator_node)
    first_node.append(0)
    target_nodes_lst.append(first_node)
    s_lst = []
    s_lst.append(0)
    h_value = 0
    last_reward = 0
    while len(closeness) != 0:
        iner_target_node = []
        closeness = closeness_deg(main_graph)
        print('len(closeness): ', len(closeness))
        if len(closeness) == 0:
            print ('Network has disintegrated successfuly in automata')
            return conct_lst , cost, target_nodes_lst, h_table
        else:
            for node in attack_list:
                if node not in closeness:
                    index = attack_list.index(node)
                    attack_list.pop(index)
                    print('alone node hase deleted: ', node)
        active_node_lst = active_node(iner_matrix)
        last_node = target_nodes_lst[-1][0]
        last_state = s_lst[-1]
        target_node, next_reward = target_node_aut_learning(iner_matrix , active_node_lst , conct , p)
        iner_target_node.append(target_node)
        iner_target_node.append(next_reward)
        target_nodes_lst.append(iner_target_node)
        print('target_node_a:' , target_node)
        cost_internal = cost_count(main_graph, [target_node], p)
        cost= cost + cost_internal[0][1]
        print('target_node: ' , target_node , "\n",'last_reward:' , last_reward , "\n",'next_reward:' , next_reward , "\n",
              'h_table:',  h_table)
        last_node = target_nodes_lst[-1][0]
        last_state = s_lst[-1]
        s_lst.append(s_lst[-1]+1)
        current_state = s_lst[-1]
        h_table  = h_value_count_update(current_state, target_node , h_table , a)

        print('H_tableeeeee:' , h_table)
        attack_list, iner_matrix = disintegration(target_node, iner_matrix, attack_list)
        main_graph = create_main_graph_copy(iner_matrix, Label)
        closeness = closeness_deg(main_graph)
        sort_order = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
        connectivity = connectivity_count(main_graph)
        conct = (connectivity/Main_Conct)
        conct_lst.append(conct)
        browse.append(target_node)
        if len(closeness) == 0:
            print ('Network has disintegrated successfuly in automata')
            return conct_lst , cost  , target_nodes_lst, h_table
    return  conct_lst, cost  , target_nodes_lst , h_table


def automata_cost_creation( node, p):
    connctivity_aut = 0.0

    cost = []
    for i in range(len(p)):
        #internal_matrix = deepcopy(main_matrix)
        connctivity_aut_inter, cost_aut, target_nodes_lst = automata_dis(Main_Matrix ,node,  p[i] )
        cost.append(cost_aut)
        if i == 0:
            connctivity_aut = connctivity_aut_inter
    return connctivity_aut , cost


def automata_learn_episodic(total_node, episode, main_matrix, p, a):
    h_table = table_initiator_aut(total_node)
    for i in range(episode):
        conct_lst, cost, target_nodes_lst , h_table = automata_dis(main_matrix, p, a, h_table)
        print('H_Table' , h_table)
    print('data type of h_table:' , type(h_table))
    np.save('H_table.npy' , h_table)
    h_table_load = np.load('H_table.npy' )
    print('H_Table_load' , h_table_load)
    print('data type of q_table_load:' , type(h_table_load))
    return h_table



#_______________Q_Learning____________


def target_node_q_learning( q_table, last_node, last_state, last_reward,  matrix , active_lst , conct , p, landa, gama):
    #yek bar attack rooye yek node tasadofi anjam shode va bad az an parameter ha pas dade shode
    # main_Graph: geraf bad az avalin hamle
    # matrix: matrix bad az avalin hamle
    # active_list: list active node hayee ke bad az attack tasadofi moondan
    # conct: az avalin hamle mohasebe shode
    iner_matrix = deepcopy(matrix)
    internal_main_graph = create_main_graph(iner_matrix, Label)
    active = deepcopy(active_lst)
    print('len(active_lst):' , len(active_lst), "\n", 'active_list:', active_lst)
    cost = cost_count(internal_main_graph, active_lst , p) #
    reward = []
    numerate = [] # soorate kasre mohasebe reward
    for i in active_lst:
        paire = []
        iner_matrix1 = deepcopy(iner_matrix)
        list, iner_matrix1 = disintegration(i, iner_matrix1, active)
        print('00000000000000000000000')
        internal_main_graph = create_main_graph_copy(iner_matrix1, Label)
        connectivity = connectivity_count(internal_main_graph)
        inter_con = (connectivity /Main_Conct)
        subtrac = conct- inter_con
        paire.append(i)
        paire.append(subtrac)
        numerate.append(paire) #yek list az azaye dotayi sakhte mishe ke har ozv mige kodoom node ro age hamle konim soorate kasr chi mishe
        print('i:' , i , 'paire:', paire)
        print('soorate kasre reward:', numerate)
        iner_matrix = deepcopy(matrix)
    # hala ye cost darim ye list soorat baraye kasr ha
    print('cost: ', cost)
    print('active_list: ', active_lst)
    print('attack:', active)
    print('list:', list)
    if len(cost) != len(numerate):
        print('toolha yeki nist', "\n", 'len cost:', len(cost) ,'len numerate', len(numerate) )
        return
    for i in range(len(cost)):
        if cost[i][0] != numerate[i][0]:
            print('tartib hamkhani nadarad')
            return
        else:
            r = []
            temp = numerate[i][1]/cost[i][1]
            r.append(cost[i][0])
            r.append(temp)
            reward.append(r) # in list dotayi hast. shomare node va reward
    print(reward)
    q_value_lst = []
    #q(St,at) = q(St,at) + landa(rt + Gama * max Q(St+1 , a) - Q(St , at))
    for n in reward:
        internal_val = []
        q_table[last_state][last_node] = q_table[last_state][last_node] + landa*(last_reward+ gama*(n[1]) - q_table[last_state][last_node])
        re = q_table[last_state][last_node]
        internal_val.append(n[0])
        internal_val.append(re)
        q_value_lst.append(internal_val)

    node = []
    reward_pure = []
    for i in q_value_lst:
        node.append(i[0])
        reward_pure.append(i[1])
    data_frame = pd.DataFrame({
        "node_number" : node,
        "q_value" : reward_pure,
        })
    target_decision = []
    print(data_frame)
    column0 = data_frame["q_value"]
    max_reward = column0.max()
    target_node_p = data_frame['node_number'][data_frame[data_frame['q_value'] == max_reward].index.tolist()].tolist()
    target_decision.append(target_node_p[0])
    print('target_decision:' , target_decision)
    # del(iner_matrix)
    # del(matrix)
    return target_decision[0] , max_reward


def q_value_count_update(last_node, last_state, last_reward, current_state,  next_node ,next_reward, q_table , landa , gama):
    # q(St,at) = q(St,at) + landa(rt + Gama * max Q(St+1 , a) - Q(St , at))
    # sample: delta = 1+ 0.9*0-0 = 1
    #         0+ 0.1*1 = 0.1
    old_value = q_table[last_state][last_node]
    # print('next_node: ' , next_node , "\n" ,'next_reward:' ,  next_reward, "\n",
    #           'q_table:',  q_table , "\n", 'landa', landa ,"\n", 'gama', gama)
    new_value = old_value + (landa*(last_reward+ (gama*next_reward)) - old_value)
    q_table[current_state][next_node] = next_reward
    q_table[last_state][last_node] = new_value
    print('newwwwww value: ' , new_value)
    # for i in map_lst:
    #     if i[1] == target_node:
    #         j = i[0]
    #         print ('i: ', i , 'j: ', j)
    #         q_table[i[0]][j] = new_q
    print('q_table after update: ' ,q_table)
    return q_table, new_value


def q_learning(main_matrix, p , landa , gama, q_table):
    cost = 0.0
    iner_matrix = deepcopy(main_matrix)
    main_graph = create_main_graph_copy(iner_matrix, Label)
    attack_list = []
    initiator_node = rand_node(main_graph)
    #q_table = table_initiator(Total_Node)
    attack_list, iner_matrix = disintegration(initiator_node, iner_matrix, attack_list)
    print('attack lst after Q_learning init: ' , attack_list)
    main_graph = create_main_graph_copy(iner_matrix, Label)
    closeness = closeness_deg(main_graph)
    conct_lst = []
    conct_lst.append(1)
    connectivity = connectivity_count(main_graph)
    conct = (connectivity/Main_Conct)
    print('first connectivity: ', conct)
    conct_lst.append(conct)
    browse = []
    browse.append(initiator_node)
    #list of browsed nodes for disintegrating
    target_nodes_lst = []
    first_node = []
    first_node.append(initiator_node)
    first_node.append(0)
    target_nodes_lst.append(first_node)
    s_lst = []
    s_lst.append(0)
    q_value = 0
    last_reward = 0
    while len(closeness) != 0:
        iner_target_node = []
        closeness = closeness_deg(main_graph)
        print('len(closeness): ', len(closeness))
        if len(closeness) == 0:
            print ('Network has disintegrated successfuly in Q_learning')
            return conct_lst , cost , q_value, target_nodes_lst , q_table
        else:
            for node in attack_list:
                if node not in closeness:
                    index = attack_list.index(node)
                    attack_list.pop(index)
                    print('alone node hase deleted: ', node)
        active_node_lst = active_node(iner_matrix)
        last_node = target_nodes_lst[-1][0]
        last_state = s_lst[-1]
        next_node, next_reward = target_node_q_learning(q_table, last_node, last_state, last_reward,
                                                       iner_matrix , active_node_lst , conct , p, landa, gama)
        iner_target_node.append(next_node)
        iner_target_node.append(next_reward)
        target_nodes_lst.append(iner_target_node)
        print('target_node_a:' , next_node)
        cost_internal = cost_count(main_graph, [next_node], p)
        cost= cost + cost_internal[0][1]
        print('next_node: ' , next_node , "\n",'last_reward:' , last_reward , "\n",'next_reward:' , next_reward , "\n",
              'q_table:',  q_table , "\n", 'landa', landa ,"\n", 'gama', gama)
        last_node = target_nodes_lst[-1][0]
        last_state = s_lst[-1]
        s_lst.append(s_lst[-1]+1)
        current_state = s_lst[-1]
        q_table , q_value_internal  = q_value_count_update(last_node, last_state, last_reward, current_state ,
                                                           next_node ,next_reward, q_table , landa , gama)
        last_reward = next_reward
        q_value = q_value + q_value_internal
        attack_list, iner_matrix = disintegration(next_node, iner_matrix, attack_list)
        main_graph = create_main_graph_copy(iner_matrix, Label)
        closeness = closeness_deg(main_graph)
        sort_order = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
        connectivity = connectivity_count(main_graph)
        conct = (connectivity/Main_Conct)
        conct_lst.append(conct)
        browse.append(next_node)
        if len(closeness) == 0:
            print ('Network has disintegrated successfuly in Q_learning')
            return conct_lst , cost , q_value , target_nodes_lst, q_table
    return  conct_lst, cost , q_value , target_nodes_lst , q_table


def q_learning_episodic(total_node , episode, main_matrix, p, landa, gama):
    q_table = table_initiator_Q(total_node)



    for i in range(episode):
        conct_lst, cost , q_value , target_nodes_lst , q_table = q_learning(main_matrix, p, landa, gama, q_table)
    print('Q_Table' , q_table)
    print('data type of q_table:' , type(q_table))
    np.save('Q_table.npy' , q_table)
    q_table_load = np.load('Q_table.npy' )
    print('Q_Table_load' , q_table_load)
    print('data type of q_table_load:' , type(q_table_load))
    return q_table


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



# def Q_cost_creation(p, landa , gama, episode):
#
#     cost = []
#     for i in range(len(p)):
#         #internal_matrix = deepcopy(main_matrix)
#         q_table , cost_ = q_learning_episodic(Main_Matrix,  p[i] , landa, gama , episode)
#         cost.append(cost_internal)
#         # if i == 0:
#         #     connctivity_Q = connctivity_Q_inter
#     return  cost

# main
list_node_initial , Layen_Count = list_node()
print ('list_node_initial:' , list_node_initial)
Total_Matrix = create_matrix(list_node_initial)
List_Struct= list_struc(list_node_initial)
comb_dis = create_comb_array(list_node_initial)
list_of_nodes , Label = Create_List_of_Nodes(List_Struct)
print('list_of_nodes:' , list_of_nodes)
Map_dic, Total_Node = node_Mapping(list_of_nodes)
print('Map_dic:' , Map_dic)
print('Total_node:' , Total_Node)
Attack_Nodes = random_atthck_nodes(list_of_nodes)
Attack_Map = attack_maping(Attack_Nodes, Map_dic)
print('attack_map::::::::::', Attack_Map)
Main_Matrix = create_major_matrix(Total_Matrix , Layen_Count)
Main_Graph = create_main_graph(Main_Matrix, Label)
Main_Conct = connectivity_count(Main_Graph)

# Rand_Node = rand_node(Main_Graph)
# Connectivity_BTW, Cost_BTW = closeness_dis(1, Main_Matrix)
# print('cost_btw:' , Cost_BTW)
# Connectivity_DEG, Cost_DEG = closeness_dis(2, Main_Matrix)
# print('cost_DEG:' , Cost_DEG)
# Connectivity_Random , Cost_Rand = random_recursive_dis(Main_Matrix)
# print('cost_Rand:' , Cost_Rand)
Primitive_Weight_Avrg, Primitive_List_of_Weight, Connectivity_Weight, Cost_Weight = weight_recursive_dis(Main_Matrix)
print('cost_weight:' , Cost_Weight)
Connectivity_GA, Cost_GA = GA_dis(Main_Matrix , Primitive_Weight_Avrg , Primitive_List_of_Weight , 0.9, 0.05, 5)
print('cost_GA:' , Cost_GA)
# Connectivity_Greedy, Cost_Greedy = Greedy_disintegration(Main_Matrix, Map_dic, Primitive_Weight_Avrg, Primitive_List_of_Weight)
# print('cost_greedy:', Cost_Greedy)
# Connctivity_aut, Cost_aut = automata_cost_creation(  [0.0, 0.5, 1.0, 1.5, 2.0])
# print('cost_aut' , Cost_aut)
# Connctivity_Q, Cost_Q = Q_cost_creation( [0.0, 0.5, 1.0, 1.5, 2.0], 0.1, 0.9)
# print('cost_Q' , Cost_Q)
#

#plot_connect(Connectivity_Random, Connectivity_DEG, Connectivity_BTW, Connectivity_Weight, Connectivity_Greedy, Connectivity_GA, Connctivity_Q ,Connctivity_aut)
#table_view(Cost_BTW, Cost_DEG, Cost_Rand, Cost_Weight, Cost_GA, Cost_Greedy, Cost_Q, Cost_aut)




#plot_connect(con_rand, con_DC, con_BC, con_UW, con_Greedy,con_GA , con_Q ,con_DSA)

#Connctivity_Q, Cost_q , Q_value, Target_Node_Lst_Q = q_learning(Main_Matrix , 0.0 , 0.1 , 0.9)
#print('Connctivity_q:' , Connctivity_Q,'Cost_q:',  Cost_q ,'Q_value:',  Q_value)

#Q_Table = q_learning_episodic(Total_Node , 1, Main_Matrix, 1, 0.1, 0.9)

#H_Table = automata_learn_episodic(Total_Node, 2, Main_Matrix, 1, 0.2)

# conct_lst, cost, Target_Node_Lst_AUT = automata_dis(Main_Matrix,Rand_Node, 0.0)
# print('Target_Node_Lst_Q: ' , Target_Node_Lst_Q)
# print ('Target_Node_Lst_AUT: ', Target_Node_Lst_AUT)




