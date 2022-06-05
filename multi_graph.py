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
import matplotlib.pyplot as plt
from copy import copy , deepcopy
from sklearn.preprocessing import normalize
from sklearn import preprocessing
import pandas as pd
from operator import itemgetter





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
        node = np.random.randint(10, 30)
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


def random_atthck_nodes(list_of_nodes ):
    #print('len(list_of_nodes):',len(list_of_nodes))
    attacked_number = math.floor(len(list_of_nodes)/4)
    #print('attacken number on nodes:', attacked_number)
    attacked_list = random.sample(list_of_nodes, attacked_number)
    #print('attacked nodes:', attacked_list)
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
    nx.draw(gr, node_size=500,  with_labels=True)
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
    nx.draw(gr, node_size=500,  with_labels=True)
    plt.show()
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
    print('in disintegrate:')
    print (main_matrix , "\n", 'Node for attack: ', node)
    print('Total_Node:', Total_Node, "\n", 'attack_list:',attack_list)
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
    return final_attack_list , main_matrix


def closeness_dis(type , main_matrix):
    # aval bayad ye peygham neshoon bedim ke in che noe disi hast
    iner_matrix = deepcopy(main_matrix)
    main_graph = create_main_graph_copy(iner_matrix, Label)
    connectivity_lst = []
    #attack_list = Attack_Map
    attack_list = []
    switcher={
                1: closeness_btw(main_graph),
                2: closeness_deg(main_graph),
                }
    closeness = switcher.get(type,"Invalid type")
    #print('closeness type------', closeness)
    #print('attack_list recurmmmmmmmmm', attack_list)
    while len(closeness) != 0:
        switcher={
                1: closeness_btw(main_graph),
                2: closeness_deg(main_graph),
                }
        closeness = switcher.get(type,"Invalid type")
        #print('closeness before sorting: ', closeness)
        sort_order = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
        #print ('sorted:::', sort_order)
        if len(closeness) == 0:
            print('final main matrix for other methodes: ', Main_Matrix)
            print ('Network has disintegrated successfuly')
            return connectivity_lst
        else:
            if len(closeness) != 0 and len(attack_list)==0:
                max_order_node = sort_order[0][0]
            for node in attack_list:
                if node not in closeness:
                    index = attack_list.index(node)
                    attack_list.pop(index)
                    print('alone node hase deleted: ', node)
            #sort_order , max_order, attack_list_rand = attack_Node_Ordering(attack_list, closeness )
            max_order_node = sort_order[0][0]
            print('target node: ', max_order_node)
            attack_list , iner_matrix = disintegration(max_order_node, iner_matrix, attack_list)
            print ('iner_matrix in closeness recursive dis:', "\n", iner_matrix)
            main_graph = create_main_graph_copy(iner_matrix, Label)
            connctivity = connectivity_count(main_graph)
            print('connectivity', connctivity)
            connectivity_lst.append(connctivity)
            print('connectivity_lst', connectivity_lst)


def random_recursive_dis(main_matrix):
     iner_matrix = deepcopy(main_matrix)
     main_graph = create_main_graph(iner_matrix, Label)
     connectivity_lst = []
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
            return connectivity_lst
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
            attack_list , iner_matrix = disintegration(rand_order_node, iner_matrix, attack_list)
            main_graph = create_main_graph(iner_matrix, Label)
            connectivity = connectivity_count(main_graph)
            connectivity_lst.append(connectivity)


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
     iner_main_matrix = deepcopy(main_matrix)
     main_graph = create_main_graph_copy(iner_main_matrix, Label)
     connectivity_lst = []
     attack_list = []
     list_of_weight  = weight_def (iner_main_matrix)
     primitive_list_of_weight = deepcopy(list_of_weight)
     active_nodes = active_node(iner_main_matrix)
     node_averg = weight_account(list_of_weight, active_nodes)
     primitive_node_avrg = deepcopy(node_averg)
     attack_list.append(node_averg[0][1])
     while len(active_nodes) != 0:
        for node in attack_list:
            if node not in active_nodes:
                index = attack_list.index(node)
                attack_list.pop(index)
                print('alone node hase deleted: ', node)
        if len(attack_list) != 0:
            target_node = attack_list[0]
        else:
            if len(active_nodes)!= 0 and len(attack_list)==0:
            # baraye jologiri az khata ya loop binahayat da moredi ke graph az aval chand bakhshi boode
                main_graph = create_main_graph_copy(iner_main_matrix, Label)
                list_of_weight  = weight_def (iner_main_matrix)
                primitive_list_of_weight = deepcopy(list_of_weight)
                node_averg = weight_account(list_of_weight, active_nodes)
                primitive_node_avrg = deepcopy(node_averg)
                attack_list.append(node_averg[0][1])
        #print('target node: ', target_node)
        attack_list , iner_main_matrix = disintegration(target_node, iner_main_matrix, attack_list)
        main_graph = create_main_graph_copy(iner_main_matrix, Label)
        connectivity = connectivity_count(main_graph)
        connectivity_lst.append(connectivity)
        #print('attack_list', attack_list)
        #list_of_weight  = weight_def (iner_main_matrix)
        active_nodes = active_node(iner_main_matrix)
        node_averg = weight_account(list_of_weight, active_nodes)
        attack_list = attack_weight_sort(attack_list , node_averg)
     if len(active_nodes) == 0:
            print ('Network has disintegrated successfuly by wight method ')
            return primitive_node_avrg , primitive_list_of_weight , connectivity_lst


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


def parent_choose(bc, dc, uw):

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
            create_dataset = False
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
        print (data_frame)
        column = data_frame["sum"]
        max_sum_value = column.max()
        target_node = data_frame['node_number'][data_frame[data_frame['sum'] == max_sum_value].index.tolist()].tolist()
        #indx = data_frame.loc[data_frame['sum'] == max_sum_value, index]
        print('max_sum_value:' , max_sum_value , 'target_node' , target_node)

    return target_node[0]


def GA_dis (main_matrix , attack_list, primitive_averg_weight_duble, primitive_weight_triple):
    attack_lst = deepcopy(attack_list)
    iner_matrix = deepcopy(main_matrix)
    main_graph = create_main_graph(iner_matrix, Label)
    connectivity_lst = []
    weight_list_avrg = deepcopy(primitive_averg_weight_duble)
    weight_list_triple = deepcopy(primitive_weight_triple)
    bc = closeness_btw(main_graph)
    dc = closeness_deg(main_graph)
    bc_sort = sorted(bc.items(), key=lambda x: x[1], reverse=True)
    dc_sort = sorted(dc.items(), key=lambda x: x[1], reverse=True)
    #weight_list_avrg = weight_account_copy(weight_list_triple , attack_list)
    print('weight: ', weight_list_avrg, "\n", 'weight_list_triple: ',weight_list_triple ,  "\n", 'bc: ', bc_sort ,"\n",  'dc', dc_sort , "\n", 'attack_lst:', attack_lst)
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
    attack_weight = []
    attack_bc = []
    attack_dc = []
    #create attack list from bc dc uw
    for n in attack_lst:
        for node in weight_normal:
            if n == node[0]:
                attack_weight.append(node)
    for n in attack_lst:
        for node in bc_normal:
            if n == node[0]:
                attack_bc.append(node)
    for n in attack_lst:
        for node in dc_normal:
            if n == node[0]:
                attack_dc.append(node)
    active_nodes = active_node(iner_matrix)
    target_node = parent_choose(attack_bc, attack_dc, attack_weight)

    while len(active_nodes) != 0:
        print('active_nodes in do while:',active_nodes)
        print ('attack_lst in do while list:',attack_lst)
        if len(active_nodes) == 0:
            print ('Network has disintegrated successfuly in GA')
            return
        else:
            if len(active_nodes)!= 0 and len(attack_list)==0:
                    main_graph = create_main_graph(iner_matrix, Label)
                    weight_list_avrg = deepcopy(primitive_averg_weight_duble)
                    weight_list_triple = deepcopy(primitive_weight_triple)
                    bc = closeness_btw(main_graph)
                    dc = closeness_deg(main_graph)
                    bc_sort = sorted(bc.items(), key=lambda x: x[1], reverse=True)
                    dc_sort = sorted(dc.items(), key=lambda x: x[1], reverse=True)
                    print('weight: ', weight_list_avrg, "\n", 'weight_list_triple: ',weight_list_triple ,  "\n", 'bc: ', bc_sort ,"\n",  'dc', dc_sort , "\n", 'attack_lst:', attack_lst)
                    weight_list_reverse = []
                    for n in weight_list_avrg:
                        temp_n = []
                        temp_n.append(n[1])
                        temp_n.append(n[0])
                        weight_list_reverse.append(temp_n)
                    weight_normal = normalize(weight_list_reverse)
                    bc_normal = normalize(bc_sort)
                    dc_normal = normalize(dc_sort)
                    attack_weight = []
                    attack_bc = []
                    attack_dc = []
                    for n in attack_lst:
                        for node in weight_normal:
                            if n == node[0]:
                                attack_weight.append(node)
                    for n in attack_lst:
                        for node in bc_normal:
                            if n == node[0]:
                                attack_bc.append(node)
                    for n in attack_lst:
                        for node in dc_normal:
                            if n == node[0]:
                                attack_dc.append(node)
                    active_nodes = active_node(iner_matrix)
                    target_node = parent_choose(attack_bc, attack_dc, attack_weight)
            for node in attack_lst:
                if node not in active_nodes:
                    index = attack_lst.index(node)
                    attack_lst.pop(index)
                    print('alone node hase deleted: ', node)
            attack_lst, iner_matrix = disintegration(target_node, iner_matrix, attack_lst)
            #print('attack_lst after disintegration :', attack_lst)
            active_nodes = active_node(iner_matrix)
            main_graph = create_main_graph(iner_matrix, Label)
            connectivity = connectivity_count(main_graph)
            connectivity_lst.append(connectivity)
            if len(active_nodes)== 0:
                print ('Network has disintegrated successfuly in GA')
                return connectivity_lst
            #main_graph = create_main_graph(iner_matrix, Label)
            bc = closeness_btw(main_graph)
            dc = closeness_deg(main_graph)
            bc_sort = sorted(bc.items(), key=lambda x: x[1], reverse=True)
            dc_sort = sorted(dc.items(), key=lambda x: x[1], reverse=True)
            # dobare vazn nodha ro hesab mikonim
            print('weight_list_reverse:  in recursive: ' , weight_list_reverse , "\n", 'active_nodes in recursive:' , active_nodes)
            node_averg = weight_account_copy(weight_list_triple, active_nodes)
            #print('len(node_averg): --------------', len(node_averg) , 'node_averg:', node_averg)
            attack_weight = attack_weight_sort_copy(attack_lst , node_averg )
            weight_list_reverse = []
            for n in node_averg:
                temp_n = []
                temp_n.append(n[1])
                temp_n.append(n[0])
                weight_list_reverse.append(temp_n)
            #print('len(weight_list_reverse): -----------------', len(weight_list_reverse), 'weight_list_reverse in do wjile'  , weight_list_reverse)

            weight_normal = normalize(weight_list_reverse)
            bc_normal = normalize(bc_sort)
            dc_normal = normalize(dc_sort)
            #motabeghe node haye attack az bd , dc , weight mikeshe biron
            attack_weight = []
            attack_bc = []
            attack_dc = []
            for n in attack_lst:
                for node in weight_normal:
                    if n == node[0]:
                        attack_weight.append(node)
            for n in attack_lst:
                for node in bc_normal:
                    if n == node[0]:
                        attack_bc.append(node)
            for n in attack_lst:
                for node in dc_normal:
                    if n == node[0]:
                        attack_dc.append(node)
            #print('weight: ', attack_weight, "\n", 'bc: ', attack_bc ,"\n",  'dc', attack_dc , "\n", 'attack_lst:', attack_lst)
            target_node = parent_choose(attack_bc, attack_dc, attack_weight)
            print('target_node in last step of dis : ',target_node)


def Greedy_disintegration (main_matrix, map_dic , primitive_averg_weight_duble, primitive_weight_triple):
    iner_matrix = deepcopy(main_matrix)
    main_graph = create_main_graph_copy(iner_matrix, Label)
    connectivity_lst = []
    weight_list_avrg = deepcopy(primitive_averg_weight_duble)
    weight_list_triple = deepcopy(primitive_weight_triple)
    bc = closeness_btw(main_graph)
    dc = closeness_deg(main_graph)
    bc_sort = sorted(bc.items(), key=lambda x: x[1], reverse=True)
    dc_sort = sorted(dc.items(), key=lambda x: x[1], reverse=True)
    print('len(primitive_averg_weight_duble)', len(primitive_averg_weight_duble))
    print(' len(primitive_weight_triple)',  len(primitive_weight_triple))
    print('len(map_dic)',len(map_dic))
    print('bc_sort:', len(bc_sort))
    print('dc_sort' , len(dc_sort))
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
    target_node = parent_choose(bc_normal, dc_normal, weight_normal)
    print('target_node:', target_node)
    while len(active_nodes) != 0:
         print('active_nodes in do while:',active_nodes)
         if len(active_nodes) == 0:
             print('Network has disintegrated successfuly in Greedy')
             return connectivity_lst
         else:
             if len(active_nodes)!=0 and len (attack_lst) ==0 :
                    weight_list_avrg = deepcopy(primitive_averg_weight_duble)
                    weight_list_triple = deepcopy(primitive_weight_triple)
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
                    weight_list_reverse_sort = sorted(weight_list_reverse, key=itemgetter(0))
                    weight_normal = normalize(weight_list_reverse_sort)
                    bc_normal = normalize(bc_sort)
                    dc_normal = normalize(dc_sort)
                    active_nodes = active_node(iner_matrix)
                    bc_normal = sorted(bc_normal, key=itemgetter(0))
                    dc_normal = sorted(dc_normal, key=itemgetter(0))
                    target_node = parent_choose(bc_normal, dc_normal, weight_normal)

             attack_lst, iner_matrix = disintegration(target_node, iner_matrix, [])
             print('attack_lst after disintegration :', attack_lst)
             active_nodes = active_node(iner_matrix)
             if len(active_nodes)== 0:
                 print ('Network has disintegrated successfuly in Greedy')
                 return
             main_graph = create_main_graph_copy(iner_matrix, Label)
             connectivity = connectivity_count(main_graph)
             connectivity_lst.append(connectivity)
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
             target_node = parent_choose(bc_normal, dc_normal, weight_normal)
             print('target_node in last step of dis : ',target_node)


def cost_count(main_graph , atc_lst , p):
    # just cost counting for each member of attack list
    degree = closeness_deg(main_graph)
    degree_lst = sorted(degree.items(), key=lambda x: x[1], reverse=True)
    #p = [0, 0.5, 1, 1.5, 2]
    sum_p = 0.0
    # sum_p1 = 0.0
    # sum_p2 = 0.0
    # sum_p3 = 0.0
    # sum_p4 = 0.0
    for i in degree_lst: # makhraje kasr ro inja misazim va rooye hameye grapg hast
        sum_p = sum_p +(i[1]**p)
        # sum_p1 = sum_p1 +(i[1]**p[1])
        # sum_p2 = sum_p2 +(i[1]**p[2])
        # sum_p3 = sum_p3 +(i[1]**p[3])
        # sum_p4 = sum_p4 +(i[1]**p[4])
    cost = []
    for i in atc_lst: # soorate kasr faghat baraye azaye attack_lst
        internal_cost = []
        for j in degree_lst:
            if i == j[0]:
                cost_p = j[1]**p
                c = (cost_p/sum_p)*100 # mohasebeye kasr be ezaye p haye mokhtalef
                internal_cost.append(i) # sakhte yek list ke har ozv an yek  node az attack_lst ast va hazine ba p haye mokhtalef
                internal_cost.append(c)
                cost.append(internal_cost)


    print('cost::::' , cost)

    return cost


def target_node_automata( matrix , attack_list , conct , p):
    #yek bar attack rooye ek node tasadofi anjam shode va bad az an parameter ha pas dade shode
    # main_Graph: geraf bad az avalin hamle
    # matrix: matrix bad az avalin hamle
    # attack_list: list hamsaye haye avalin node ke be soorate tasadofi entekhb shod va attack shod
    # conct: az avalin hamle mohasebe shode
    iner_matrix = deepcopy(matrix)
    print('sssssssssssssssssss')
    internal_main_graph = create_main_graph_copy(iner_matrix, Label)
    attack = deepcopy(attack_list)
    print('len(attack_lst):' , len(attack_list), "\n", 'attack_list:', attack_list)
    cost = cost_count(internal_main_graph, attack_list , p) #
    reward = []
    numerate = [] # soorate kasre mohasebe reward
    for i in attack_list:
        paire = []
        list, iner_matrix = disintegration(i, iner_matrix, attack)
        print('00000000000000000000000')
        internal_main_graph = create_main_graph_copy(iner_matrix, Label)
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
    print('attack_list: ', attack_list)
    print('attack:', attack)
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
            p = numerate[i][1]/cost[i][1]
            # p1 = numerate[i][1]/cost[i][2]
            # p2 = numerate[i][1]/cost[i][3]
            # p3 = numerate[i][1]/cost[i][4]
            # p4 = numerate[i][1]/cost[i][5]
            r.append(cost[i][0])
            r.append(p)
            # r.append(p1)
            # r.append(p2)
            # r.append(p3)
            # r.append(p4)
            reward.append(r)
    print(reward)
    node = []
    p = []

    for i in reward:
        node.append(i[0])
        p.append(i[1])
        # p1.append(i[2])
        # p2.append(i[3])
        # p3.append(i[4])
        # p4.append(i[5])

    data_frame = pd.DataFrame({
        "node_number" : node,
        "p" : p,
        # "p1" : p1,
        # "p2" : p2,
        # "p3" : p3,
        # "p4" : p4,
        # "p4" : p4
        })
    target_decision = []
    print(data_frame)
    column0 = data_frame["p"]
    max_p_value = column0.max()
    target_node_p = data_frame['node_number'][data_frame[data_frame['p'] == max_p_value].index.tolist()].tolist()
    # column1 = data_frame["p1"]
    # max_p1_value = column1.max()
    # target_node_p1 = data_frame['node_number'][data_frame[data_frame['p1'] == max_p1_value].index.tolist()].tolist()
    # column2 = data_frame["p2"]
    # max_p2_value = column2.max()
    # target_node_p2 = data_frame['node_number'][data_frame[data_frame['p2'] == max_p2_value].index.tolist()].tolist()
    # column3 = data_frame["p3"]
    # max_p3_value = column3.max()
    # target_node_p3 = data_frame['node_number'][data_frame[data_frame['p3'] == max_p3_value].index.tolist()].tolist()
    # column4 = data_frame["p4"]
    # max_p4_value = column4.max()
    # target_node_p4 = data_frame['node_number'][data_frame[data_frame['p4'] == max_p4_value].index.tolist()].tolist()
    # print(target_node_p0 , target_node_p1 , target_node_p2, target_node_p3, target_node_p4)
    target_decision.append(target_node_p[0]) #, target_node_p1, target_node_p2, target_node_p3, target_node_p4)
    print('target_decision:' , target_decision)
    return target_decision[0]


def automata_dis(main_matrix , p):
    iner_matrix = deepcopy(main_matrix)
    main_graph = create_main_graph_copy(iner_matrix, Label)
    attack_list = []
    closeness = closeness_deg(main_graph)
    sort_order = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
    rand_order_node = sort_order[np.random.randint(0, len(sort_order))][0]
    attack_list, iner_matrix = disintegration(rand_order_node, iner_matrix, attack_list)
    main_graph = create_main_graph_copy(iner_matrix, Label)
    conct_lst = []
    connectivity = connectivity_count(main_graph)
    conct = (connectivity/Main_Conct)
    conct_lst.append(conct)
    browse = []
    browse.append(rand_order_node)
    while len(closeness) != 0:
        closeness = closeness_deg(main_graph)
        if len(closeness) == 0:
            print('final main matrix for other methodes: ', Main_Matrix)
            print ('Network has disintegrated successfuly in automata')
            return #connectivity_lst
        else:
            for node in attack_list:
                if node in closeness:
                    flag = "true"
                else:
                    index = attack_list.index(node)
                    attack_list.pop(index)
                    print('alone node hase deleted: ', node)

            target_node_a = target_node_automata( iner_matrix , attack_list , conct , p)
            print('target_node_a:' , target_node_a)
            attack_list , iner_matrix = disintegration(target_node_a, iner_matrix, attack_list)
            main_graph = create_main_graph_copy(iner_matrix, Label)
            #closeness = closeness_deg(main_graph)
            sort_order = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
            connectivity = connectivity_count(main_graph)
            conct = (connectivity/Main_Conct)
            conct_lst.append(conct)
            browse.append(target_node_a)
    return browse, conct_lst



# main
list_node_initial , Layen_Count = list_node()
Total_Matrix = create_matrix(list_node_initial)
List_Struct= list_struc(list_node_initial)
comb_dis = create_comb_array(list_node_initial)
list_of_nodes , Label = Create_List_of_Nodes(List_Struct)
Map_dic, Total_Node = node_Mapping(list_of_nodes)
Attack_Nodes = random_atthck_nodes(list_of_nodes)
Attack_Map = attack_maping(Attack_Nodes, Map_dic)
Main_Matrix = create_major_matrix(Total_Matrix , Layen_Count)
Main_Graph = create_main_graph(Main_Matrix, Label)
Main_Conct = connectivity_count(Main_Graph)
#Connectivity_BTW = closeness_dis(1, Main_Matrix)
#Connectivity_DEG = closeness_dis(2, Main_Matrix)
#Connectivity_Random = random_recursive_dis(Main_Matrix)
#Primitive_Weight_Avrg, Primitive_List_of_Weight, Connectivity_Weight = weight_recursive_dis(Main_Matrix)
#Connectivity_GA = GA_dis(Main_Matrix, Attack_Map , Primitive_Weight_Avrg , Primitive_List_of_Weight)
#Connectivity_Greedy = Greedy_disintegration(Main_Matrix, Map_dic, Primitive_Weight_Avrg, Primitive_List_of_Weight)

#automata_dis(Main_Matrix , 0.0) #[0.0, 0.5, 1.0, 1.5, 2.0]







