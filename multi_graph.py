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
        node = np.random.randint(3, 7)
        list_node.append(node)
    #print('list_node:', list_node)
    return list_node


def list_struc(list_node):
    # gets the number of layers, creates and returns struct
    struc = []
    for x in range(len(list_node)):
        str = [x, list_node[x]]
        struc.append(str)

    print('struc:', struc)
    return struc


def random_weighted_graph(n):
    """" create a random graph by n= nodes number, p = probability , lower and upper wight"""

    #print('in create weighted graph n: ', n)
    p = np.random.uniform(0.2, 1)
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
    print ('layer number:', n)
    total_mtrx = np.zeros((n, n), dtype="object", order='c')
    # total_mtrx = np.empty((n, n, n, n ), dtype="int", order='c')
    #print('total_mtrx:', total_mtrx)
    i = 0
    j = 0
    edge_list = []
    """create diagonal of main matrix """

    for l in range(n): # n = number of layers
        print ('l:', l)
        m = list_node[l]
        print ('l:' , l , 'list_node[l]: ' , m)
        adjc_mtrx = random_weighted_graph(m) # craete random weighted graph for each layer. l = number of nodes
        #adjc_mtrx = g.get_adjacency()
        print('xxxxxx:', adjc_mtrx)
        total_mtrx[i][j] = adjc_mtrx
        i = i + 1
        j = j + 1

    #create Bipartite matrixes
    comb = create_comb_array(list_node) #for each member of combarray: 0.comb, 1.B0, 2.B1, 3.B2, 4.B3, 5.R(edge_number), 6.n(lenth)
    for p in comb:
        print ('number of biprt edge:' , p[5], 'B1:', p[2] , 'B3', p[4])
        G = nx.bipartite.gnmk_random_graph(p[2], p[4], p[5])
        G_lenth =(p[6])
        Row_order = range(G_lenth)
        print ('row_order:', Row_order)
        #G_adj0 = np.zeros((G_lenth, G_lenth), order='c')
        G_adj = nx.bipartite.biadjacency_matrix(G,row_order=Row_order, column_order= Row_order)
        print('Type of G_adj0:',type(G_adj))
        print('G_adj:', G_adj)
        print ('shape:' ,csr_matrix.get_shape(G_adj))
        #G_dens = csr_matrix.todense(G_adj)
        G_array = csr_matrix.toarray(G_adj)
        print('G_array:', G_array, G_array.ndim , G_array[0], len(G_array[0]), G_array[0][0])

        dens_matrx= np.zeros((G_lenth, G_lenth), order='c')
        for i in range(G_lenth):
            for j in range(G_lenth):
                dens_matrx[i][j]= int(G_array[i][j])
        print('dens_matrx:', dens_matrx)

        for i in range(0 , G_lenth):
            for j in range(0, G_lenth):
                if dens_matrx[i][j]==1:
                    dens_matrx[j][i]= 1

        print('dens_matrx after convertion :', dens_matrx)
        total_mtrx[p[1]][p[3]] = dens_matrx
        total_mtrx[p[3]][p[1]] = dens_matrx


    print('tot:', total_mtrx)
    print(len(total_mtrx))
    return total_mtrx


def Create_List_of_Nodes(List_Struct):
    list_of_Node = []
    for n in list_Struct:
        for m in range(n[1]):
            peer = []
            peer.append(n[0])
            peer.append(m)
            list_of_Node.append(peer)
    print('list of nodes:', list_of_Node)
    return list_of_Node


def node_Mapping (list_of_Node):
    i = 0
    map_dic = {}
    for node in list_of_Node:
        map_dic[i] = node
        i = i+1
    print('**************map_dic', map_dic , 'i:', i)
    return map_dic , i


def Create_Huristic_Atthck_Nodes(list_of_nodes ):
    print('len(list_of_nodes):',len(list_of_nodes))
    attacked_number = math.floor(len(list_of_nodes)/4)
    print('attacken number on nodes:', attacked_number)
    attacked_list = random.sample(list_of_nodes, attacked_number)
    print('attacked nodes:', attacked_list)
    return attacked_list


def attacked_node_struct(attacked_nodes):
    print('---attacked_nodes which passed to attacked_node_struct:', attacked_nodes)
    node_struct = []
    for node in attacked_nodes:
        node_struct_temp = []
        layer = node[0]
        node_num = node[1]# index of attacked node in its layer
        Struct = list_Struct[layer]# get struct if the layer that belongs to attacked node
        layer_Node_Number = Struct[1]# getting Node_Number and wide of adjacency matrix
        node_struct_temp.append(layer)
        node_struct_temp.append(node_num)
        node_struct_temp.append(layer_Node_Number)
        node_struct.append(node_struct_temp)
    return node_struct #for each node in node_struct as a p: p[0]=layer, p[1]=node_num, p[2]=layer_lenth

def complex_disintegration_diagonal(attacked_node_struct, total_matrix):
    step_history = []
    print('***type attack_node_struct:', type(attacked_node_struct))
    # print('attacked_node_struct[0]:',attacked_node_struct[0],
    #       'attacked_node_struct[1]:',attacked_node_struct[1],
    #       'attacked_node_struct[2]:',attacked_node_struct[2],
    #        'len:', len(attacked_node_struct))
    print('-----attacked node that passed to complex_disintegration_diagonal:', attacked_node_struct)
    for ns in attacked_node_struct:
        print(ns)
        # for each node in node_struct as a p: p[0]=layer, p[1]=node_num, p[2]=layer_lenth
        print('----Attecked nodes:', ns, 'layer:', ns[0] , 'node_num:', ns[1],
              'layer_node_number: ',ns[2] )
        layer_adj = total_matrix[ns[0]][ns[0]]# fetching the adjacency matrix of layer in diagonal of Total_matrix
        print('layer_adj before disintegration:', layer_adj)
        for i in range(ns[2]): #layer_lenth = number of nodes in layer
            print('i:', i , 'node_num', ns[1], 'layer_Node_Number: ', ns[2] )
            #print ('layer_adj[i][node_num]:', layer_adj[i][ns[1]])
            #print ('layer_adj[node_num][i]:', layer_adj[ns[1]][i])
            j = layer_adj[i][ns[1]]
            print ('j:', j)
            diag_peer = []
            if j==1:
                diag_peer.append(ns[0])
                diag_peer.append(i)
                print('peer:', diag_peer)
                #layer_adj[node_stuc[1]][i] = 0
                #layer_adj[i][node_stuc[1]] = 0
                step_history.append(diag_peer)

    print('step_history of diag: ' , step_history)
    print('@@@@@@diag_dis finished')
    return step_history, total_matrix

def complex_disintegration_bipart(attacked_node_struct, total_matrix):
    # each node in node_struct as a p: p[0]=layer, p[1]=node_num, p[2]=layer_lenth
    #for each member of comb_dis: 0.comb, 1.B0, 2.B1, 3.B2, 4.B3, 5.R(edge_number), 6.n(lenth)
    print('attacked_nodes_struct which passed to bipart:' , attacked_node_struct)
    print('comb_dis:',comb_dis)
    step_history = []
    for p in attacked_node_struct:
        bipart_peer = []
        for n in comb_dis:
            print('p[0]:', p[0],  'n[1]', n[1])
            if p[0] == n[1]:# p= attacked node layer. n= a comp, n[1]: first layer in comb
                print('first condition by:','p[0]:', p[0],  'n[1]', n[1] )
                print('len(list_Struct): ', len(list_Struct))
                print('p0:', p[0],'p1: ', p[1], 'n[6]:', n[6], 'n[3]:', n[3])
                bi_matrix = total_matrix[n[1],n[3]]
                for j in range(n[6]):
                    peer_temp = []
                    print('n6:', n[6], 'p1:' , p[1], 'j:', j ,'n2', n[2])
                    if bi_matrix[j][p[1]]==1:
                        print ('is 1')
                        peer_temp.append(n[3])
                        if j > (n[2]-1):
                                a = j-n[2]
                                peer_temp.append(a) # dar sakhte peer bayad j ro B0 , B3 moghayese konim va order dorost ro barash mohasebe konim
                                #bi_matrix[j][p[1]] = 0
                                #bi_matrix[p[1]][j] = 0
                        else:
                                peer_temp.append(j)

                        print('peer_temp:', peer_temp)
                        step_history.append(peer_temp)
            else:
                if p[0] == n[3]:
                    print('secound condition by:', 'p[0]:', p[0],  'n[3]', n[3])
                    print('len(list_Struct): ', len(list_Struct))
                    print('p0:', p[0],'p1: ', p[1], 'n[6]:', n[6], 'n[3]:', n[3])
                    bi_matrix = total_matrix[n[1],n[3]]
                    for j in range(n[6]):
                        peer_temp = []
                        print('n6:', n[6], 'p1:' , p[1], 'j:', j, 'n4', n[4])
                        index = p[1]+(n[2]-1)
                        if bi_matrix[j][index]==1:
                            print ('is 1')
                            peer_temp.append(n[1])
                            if j > (n[4]-1):
                                j = index
                                peer_temp.append(j) # dar sakhte peer bayad j ro B0 , B3 moghayese konim va order dorost ro barash mohasebe konim
                                #bi_matrix[j][p[1]] = 0
                                #bi_matrix[p[1]][j] = 0
                            else:
                                peer_temp.append(j)
                            print('peer_temp:', peer_temp)
                            step_history.append(peer_temp)





    return step_history, total_matrix


def delete_redundant(step_dic, new_list, step):
    is_in = False
    list_recall = []
    print ('____steps: ', step_dic)
    print('---new_list: ', new_list)
    print('step: ', step)
    step_temp = []
    for s in range(step):
        step_temp = step_dic[s]
        print('____step_temp: ', step_temp)

        for n in new_list:
            if n in step_temp:
                new_list.remove(n)
        list_recall = new_list
        print('__Final list: ', new_list)


    return list_recall


def complex_disintegrate(attacked_nodes , totla_matrix):
    print('attecked_Nodes_in_dis********: ', attacked_nodes)
    step_history = {}
    step = 0
    step_history[step] = attacked_nodes
    attacked_node_step = attacked_nodes
    while attacked_node_step != []:

        print('attacked_node_step' ,attacked_node_step)
        print ('--------------------------------attacked_nodes: ', len( attacked_nodes))

        attacked_node_temp = []
        attacked_node_struc = attacked_node_struct(attacked_node_step)
        step_node_diag , totla_matrix_internal = complex_disintegration_diagonal(attacked_node_struc, totla_matrix)
        for p in step_node_diag:
            attacked_node_temp.append(p)

        step_node_bipart,totla_matrix_final= complex_disintegration_bipart(attacked_node_struc, totla_matrix)
        for p in step_node_bipart:
            attacked_node_temp.append(p)

        step = step+1
        attacked_node_step = delete_redundant(step_history, attacked_node_temp, step)

        step_history[step]= attacked_node_step
        print('step_history: ', step_history)
        #print('totla_matrix_final: ', totla_matrix_final)
        print(len(attacked_node_step))
    return step_history , totla_matrix_final


def create_major_matrix():


# main
list_node_initial = list_node()
Total_Matrix = create_matrix(list_node_initial)
list_Struct= list_struc(list_node_initial)
comb_dis = create_comb_array(list_node_initial)
list_of_nodes = Create_List_of_Nodes(list_Struct)
Map_dic, Total_Node = node_Mapping(list_of_nodes)
Huristic_Atthck_Nodes = Create_Huristic_Atthck_Nodes(list_of_nodes)
complex_disintegrate(Huristic_Atthck_Nodes, Total_Matrix)








