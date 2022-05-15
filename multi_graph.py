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


def complex_disintegration_diagonal(attacked_node_struct, total_matrix):
    step_history = []
    #print('***type attack_node_struct:', type(attacked_node_struct))
    # print('attacked_node_struct[0]:',attacked_node_struct[0],
    #       'attacked_node_struct[1]:',attacked_node_struct[1],
    #       'attacked_node_struct[2]:',attacked_node_struct[2],
    #        'len:', len(attacked_node_struct))
    #print('-----attacked node that passed to complex_disintegration_diagonal:', attacked_node_struct)
    for ns in attacked_node_struct:
        #print(ns)
        # for each node in node_struct as a p: p[0]=layer, p[1]=node_num, p[2]=layer_lenth
        #print('----Attecked nodes:', ns, 'layer:', ns[0] , 'node_num:', ns[1],
             # 'layer_node_number: ',ns[2] )
        layer_adj = total_matrix[ns[0]][ns[0]]# fetching the adjacency matrix of layer in diagonal of Total_matrix
        #print('layer_adj before disintegration:', layer_adj)
        for i in range(ns[2]): #layer_lenth = number of nodes in layer
            #print('i:', i , 'node_num', ns[1], 'layer_Node_Number: ', ns[2] )
            #print ('layer_adj[i][node_num]:', layer_adj[i][ns[1]])
            #print ('layer_adj[node_num][i]:', layer_adj[ns[1]][i])
            j = layer_adj[i][ns[1]]
            #print ('j:', j)
            diag_peer = []
            if j==1:
                diag_peer.append(ns[0])
                diag_peer.append(i)
                #print('peer:', diag_peer)
                #layer_adj[node_stuc[1]][i] = 0
                #layer_adj[i][node_stuc[1]] = 0
                step_history.append(diag_peer)

    # print('step_history of diag: ' , step_history)
    # print('@@@@@@diag_dis finished')
    return step_history, total_matrix


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



# def attack_Node_Ordering(attack_list:list, base_list):
#     # attack_list: list of nodes which are attacked and mapped
#     # base_list : BTW or DEG, which determines criterion of ordering
#     # list ro bar asase meyare morede barresi moratab mikone
#     if len(base_list)!= 0:
#         order_dic = {}
#         for node in base_list:
#             order_dic[node] = base_list[node]
#         sort_order = sorted(order_dic.items(), key=lambda x: x[1], reverse=True)
#         return sort_order, sort_order[0], attack_list[np.random.randint(0, len(attack_list))]
#     else:
#         return [], [], []



def disintegration (node, main_matrix, attack_list):
    # node: the first index of attack node list
    # main_matrix: matrix should updated by each disintegration step
    # attack_List: should update by each disintegration step
    # type = 1: Random / type = 2: DEG / Type = 3: BWN / Type = 4: WGHT

    neigh = []
    for i in range(Total_Node):
        if main_matrix[i][node] == 1:
            neigh.append(i)
            main_matrix[i][node] = 0
            main_matrix[node][i] = 0
    #index = attack_list.index(node)
    #attack_list.pop(index)

    for n in neigh:
        attack_list.append(n)
    final_attack_list = []
    final_attack_list = list(set(attack_list))
    return final_attack_list , main_matrix



def closeness_recursive_dis(type , main_matrix):
    # aval bayad ye peygham neshoon bedim ke in che noe disi hast
    iner_matrix = deepcopy(main_matrix)
    main_graph = create_main_graph(iner_matrix, Label)
    #attack_list = Attack_Map
    attack_list = []
    switcher={
                1: closeness_btw(main_graph),
                2: closeness_deg(main_graph),
                }
    closeness = switcher.get(type,"Invalid type")
    print('closeness type------', closeness)
    print('attack_list recurmmmmmmmmm', attack_list)
    while len(closeness) != 0:
        switcher={
                1: closeness_btw(main_graph),
                2: closeness_deg(main_graph),
                }
        closeness = switcher.get(type,"Invalid type")
        print('closeness before sorting: ', closeness)
        sort_order = sorted(closeness.items(), key=lambda x: x[1], reverse=True)

        print ('sorted:::', sort_order)
        if len(closeness) == 0:
            print('final main matrix for other methodes: ', Main_Matrix)
            print ('Network has disintegrated successfuly')
            return
        else:
            for node in attack_list:
                if node not in closeness:
                    index = attack_list.index(node)
                    attack_list.pop(index)
                    print('alone node hase deleted: ', node)
            #sort_order , max_order, attack_list_rand = attack_Node_Ordering(attack_list, closeness )
            max_order_node = sort_order[0][0]

            print('target node: ', max_order_node)
            attack_list , iner_matrix = disintegration(max_order_node, iner_matrix, attack_list)
            main_graph = create_main_graph(iner_matrix, Label)


def random_recursive_dis(main_matrix):
     iner_matrix = deepcopy(main_matrix)
     main_graph = create_main_graph(iner_matrix, Label)
    #attack_list = Attack_Map
     attack_list = []
     closeness = closeness_deg(main_graph)
     print('closeness type------', closeness)
     print('attack_list recurmmmmmmmmm', attack_list)
     while len(closeness) != 0:
        closeness =  closeness_deg(main_graph)
        print('closeness before sorting: ', closeness)
        sort_order = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
        print ('sorted:::', sort_order)
        if len(closeness) == 0:
            print('final main matrix for other methodes: ', Main_Matrix)
            print ('Network has disintegrated successfuly')
            return
        else:
            for node in attack_list:
                if node in closeness:
                    flag = "true"
                else:
                    index = attack_list.index(node)
                    attack_list.pop(index)
                    print('alone node hase deleted: ', node)
            #sort_order , max_order, attack_list_rand = attack_Node_Ordering(attack_list, closeness )

            rand_order_node = sort_order[np.random.randint(0, len(sort_order))][0]

            print('target node: ', rand_order_node)
            attack_list , iner_matrix = disintegration(rand_order_node, iner_matrix, attack_list)
            main_graph = create_main_graph(iner_matrix, Label)


def weight_def (main_matrix):
    # be ezaye ha node ye vazne tasadofi ijad mikone va liste node haye faal ro ham tashkhis mide va barmigardoone
    list_of_weight = []
    pair = []
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
    for node in list_of_weight:
        i = node[0]
        j = node[1]
        for n in list_of_weight:
            if n[0] == j and n[1] == i:
                n[2] = node[2]
    #print('list of weight correct:: ' , list_of_weight)
    active_node_list = []
    for node in list_of_weight:
        if node[0] not in active_node_list:
            active_node_list.append(node[0])
    #print('active_node: ', active_node_list)
    return list_of_weight , active_node_list


def average_count(list_node):
    #print ('list_node_out: ', list_node)
    count = 0
    sum = 0
    for node in list_node:
        count = count+1
        sum = sum + node[2]
    #print('node: ', node, 'count:', count , 'sum:', sum )
    weight_avr = (sum/count)
    #print(sum , '/', count, '=' , weight_avr)

    return weight_avr


def weight_account(list_of_weight , active_nodes ):
    node_and_avr_list = []

    for i in active_nodes:
        list_node_internal = []
        node_and_avr_temp = []
        for node in list_of_weight:
            if i == node[0]:
                list_node_internal.append(node)
        #print('list_node_internal:',list_node_internal)
        node_avr = average_count(list_node_internal)
        node_and_avr_temp.append(node_avr)
        node_and_avr_temp.append(i)
        node_and_avr_list.append(node_and_avr_temp)

    node_and_avr_list.sort()
        #= sorted(node_and_avr_list)
    node_and_avr_list.reverse()
    #print('node_and_avr_list:',node_and_avr_list)
    return node_and_avr_list


def attack_weight_sort(attack_node , node_avrg):
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


def weight_recursive_dis(main_matrix):
     #iner_main_matrix = [row[:] for row in main_matrix]
     iner_main_matrix = deepcopy(main_matrix)
     #print('iner_main_matrix' , iner_main_matrix)
     main_graph = create_main_graph(iner_main_matrix, Label)
     attack_list = []
     list_of_weight , active_nodes = weight_def (iner_main_matrix)
     node_averg = weight_account(list_of_weight, active_nodes)
     primitive_node_avrg = node_averg
     #attack_list.append(node_averg[0][1])
     #print('attack_list : ',attack_list)
     attack_list.append(node_averg[0][1])
     #print('attack_list : ',attack_list)
     while len(active_nodes) != 0:
        for node in attack_list:
            if node not in active_nodes:
                index = attack_list.index(node)
                attack_list.pop(index)
                print('alone node hase deleted: ', node)

        target_node = attack_list[0]
        #print('target node: ', target_node)
        attack_list , iner_main_matrix = disintegration(target_node, iner_main_matrix, attack_list)
        #main_graph = show_main_graph(iner_main_matrix, Label)
        #print('attack_list', attack_list)
        list_of_weight , active_nodes = weight_def (iner_main_matrix)
        node_averg = weight_account(list_of_weight, active_nodes)
        attack_list = attack_weight_sort(attack_list , node_averg)
     if len(active_nodes) == 0:
            print ('Network has disintegrated successfuly by wight method ')
            return primitive_node_avrg


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
        tmp = (value - min_value) / (max_value - min_value)
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

    return target_node


def GA_disintegration (main_matrix , attack_list, primitive_weight):
    iner_matrix = deepcopy(main_matrix)
    main_graph = create_main_graph(iner_matrix, Label)
    weight_list = deepcopy(primitive_weight)
    bc = closeness_btw(main_graph)
    dc = closeness_deg(main_graph)
    bc_sort = sorted(bc.items(), key=lambda x: x[1], reverse=True)
    dc_sort = sorted(dc.items(), key=lambda x: x[1], reverse=True)
    #     print ('sorted:::', sort_order)
    attack_lst = deepcopy(attack_list)

    print('weight: ', weight_list, "\n", 'bc: ', bc_sort ,"\n",  'dc', dc_sort , "\n", 'attack_lst:', attack_lst)
    #print('attack_map: ', attack_map)
    weight_list_reverse = []
    for n in weight_list:
        temp_n = []
        temp_n.append(n[1])
        temp_n.append(n[0])
        weight_list_reverse.append(temp_n)
    print('weight_list_reverse' , weight_list_reverse)
    weight_normal = normalize(weight_list_reverse)
    bc_normal = normalize(bc_sort)
    dc_normal = normalize(dc_sort)
    attack_weight = []
    attack_bc = []
    attack_dc = []
    for n in attack_lst:
        for node in weight_normal:
            #print ('node in weight', node , 'n', n , 'node[0]', node[0])
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

    print('attack_weight: ', attack_weight )
    print('attack_dc: ', attack_dc)
    print('attack_bc: ', attack_bc)
    print('attack_lst: ', attack_lst)
    target_node = parent_choose(attack_bc, attack_dc, attack_weight)

    # while len(closeness) != 0:
    #     switcher={
    #             1: closeness_btw(main_graph),
    #             2: closeness_deg(main_graph),
    #             }
    #     closeness = switcher.get(type,"Invalid type")
    #     print('closeness before sorting: ', closeness)
    #     sort_order = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
    #
    #     print ('sorted:::', sort_order)
    #     if len(closeness) == 0:
    #         print('final main matrix for other methodes: ', Main_Matrix)
    #         print ('Network has disintegrated successfuly')
    #         return
    #     else:
    #         for node in attack_list:
    #             if node not in closeness:
    #                 index = attack_list.index(node)
    #                 attack_list.pop(index)
    #                 print('alone node hase deleted: ', node)
    #         #sort_order , max_order, attack_list_rand = attack_Node_Ordering(attack_list, closeness )
    #         max_order_node = sort_order[0][0]
    #
    #         print('target node: ', max_order_node)
    #         attack_list , iner_matrix = disintegration(max_order_node, iner_matrix, attack_list)
    #         main_graph = create_main_graph(iner_matrix, Label)


    return




def cost_connectivity ():

    return



# main
list_node_initial , Layen_Count = list_node()
Total_Matrix = create_matrix(list_node_initial)
List_Struct= list_struc(list_node_initial)
comb_dis = create_comb_array(list_node_initial)
list_of_nodes , Label = Create_List_of_Nodes(List_Struct)
Map_dic, Total_Node = node_Mapping(list_of_nodes)
#print ('map_dic:' , Map_dic, 'total_node:', Total_Node)
Attack_Nodes = random_atthck_nodes(list_of_nodes)
Attack_Map = attack_maping(Attack_Nodes, Map_dic)
#complex_disintegrate(Huristic_Atthck_Nodes, Total_Matrix)
Main_Matrix = create_major_matrix(Total_Matrix , Layen_Count)
#print ('Main_Matrix_Type:', type(Main_Matrix))
#Main_Graph = show_main_graph(Main_Matrix, Label)
#Attack_Map = attack_Node_Mapping(Atthck_Nodes)


#closeness_recursive_dis(1, Main_Matrix)
#closeness_recursive_dis(2, Main_Matrix)
#random_recursive_dis(Main_Matrix)
Primitive_Weight = weight_recursive_dis(Main_Matrix)
#Main_Graph = create_main_graph(Main_Matrix, Label)
#print(Main_Matrix)

# BC = [(0,7), (1, 5), (5,10)]
# DC = [(0,6.3454874987538), (1, 8.2198093809834), (5,5.27198496)]
# UW = [(0,5.3454874987538), (1, 4.2198093809834), (5,6.27198496)]
# BC1 = normalize(BC)
# DC1 = normalize(DC)
# UW1 = normalize(UW)
#Max_value_fitness = parent_choose(BC1, DC1, UW1)

GA_disintegration(Main_Matrix, Attack_Map , Primitive_Weight)










