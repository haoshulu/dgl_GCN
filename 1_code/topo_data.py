import os.path as osp
import json

from torch_geometric.data import InMemoryDataset, Data

from torch.utils.data import DataLoader
from torch_geometric.data import Data, DataLoader
import torch

from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from decimal import Decimal
from reward_fn import compute_reward

import torch_geometric
import dgl

def loadData(path,y_select,ncomp):

    json_file = json.load(open(path + "/dataset" + "_" + str(ncomp) + ".json"))

    tmp = {}
    nn = 0

    for item in json_file:
        topo_id = str(item[-8:-2])
        param_id = str(item[-1])

        list_of_edge = json_file[item]["list_of_edge"]
        list_of_node = json_file[item]["list_of_node"]
        edge_attr = json_file[item]["edge_attr"]
        edge_attr0 = json_file[item]["edge_attr0"]
        node_attr = json_file[item]["node_attr"]


        target_vout = []
        target_eff = []
        target_rewards = []
        target_cls = []

        analytic_vout = []
        analytic_eff = []
        try:
            analytic_eff.append(float(json_file[item]["eff_analytic"]))
        except:
            analytic_eff.append(0)
        label_analytic_eff = analytic_eff
        try:
            analytic_vout.append(float(json_file[item]["vout_analytic"]) / 100)
        except:
            analytic_vout.append(0)
        label_analytic_vout = analytic_vout

        eff = json_file[item]["eff"]
        vout = abs(json_file[item]["vout"] / 100)
        reward = compute_reward(eff, vout)

        target_eff.append(eff)
        target_vout.append(vout)
        target_rewards.append(reward)
        target_cls.append(float(35 < json_file[item]["vout"] < 65))

        if y_select == 'reg_eff':
            label = target_eff

        elif y_select == 'reg_vout':
            label = target_vout

        elif y_select == 'reg_reward':
            label = target_rewards

        elif y_select == 'cls_buck':
            label = target_cls

        else:
            print("Wrong select input")
            continue

        if json_file[item]["vout"] == -1:
            continue

        if json_file[item]["eff"] < 0 or json_file[item]["eff"] > 1:
            continue

        rout = json_file[item]["rout"]
        cout = json_file[item]["cout"]
        freq = json_file[item]["freq"]
        duty_cycle = json_file[item]["duty_cycle"]

        if param_id=='0':
            reward_max=-1
        if param_id == '2':
            tmp[topo_id] = {}
            tmp[topo_id]['nodes'] = list_of_node
            tmp[topo_id]['edges'] = list_of_edge
            tmp[topo_id]['edge_attr1'] = edge_attr0
            tmp[topo_id]['edge_attr2'] = edge_attr
            tmp[topo_id]['node_attr'] = node_attr
            # if reward_max>0.1:
            #      #print(nn, reward_max)

            label = []
            label.append(reward_max)
            tmp[topo_id]['label'] = label

        if reward>=reward_max:
            label_max=target_rewards
            reward_max=reward
            vout_max=target_vout
            eff_max=target_eff


        if param_id=='4' and topo_id in tmp.keys():
            # if reward_max<0.1, delete topo


            #                print(label,nn)
            nn = nn + 1

            tmp[topo_id]['sim_eff'] = eff_max
            tmp[topo_id]['sim_vout'] = vout_max
            tmp[topo_id]['analytic_eff'] = label_analytic_eff
            tmp[topo_id]['analytic_vout'] = label_analytic_vout
            reward_max = -1

        else:

            continue

        #print(tmp)

    return tmp

def collectData(path,y_select,ncomp):

    tmp = loadData(path,y_select,ncomp)
    node_list = {}
    edge_list = {}
    node_feature1 = {}
    node_feature2 = {}
    label_feature = {}
    graph_list = []
    label_list = []
    datalist = []

    for fn in tmp:
        nodes_tmp =  tmp[fn]['nodes']
        edge_tmp = tmp[fn]['edges']
        ef_tmp1 = tmp[fn]['edge_attr1']
        ef_tmp2 = tmp[fn]['edge_attr2']
        label = tmp[fn]['label']

        node_list[fn]={}
        edge_list[fn]=[]
        node_feature1[fn]=[]
        node_feature2[fn]=[]
        nf_tmp_list = []
        count = 3
        for node in nodes_tmp:

            if node == "VIN":
                node_list[fn]["VIN"] = 0
                nf_tmp_list = [1, 0, 0, 0, 0, 0, 0, 0]
                node_feature1[fn].insert(0,nf_tmp_list)
                node_feature2[fn].insert(0, nf_tmp_list)
            elif node == "VOUT":
                node_list[fn]["VOUT"] = 1
                nf_tmp_list = [0, 1, 0, 0, 0, 0, 0, 0]
                node_feature1[fn].insert(1, nf_tmp_list)
                node_feature2[fn].insert(1, nf_tmp_list)
            elif node == "GND":
                node_list[fn]["GND"] = 2
                nf_tmp_list = [0, 0, 1, 0, 0, 0, 0, 0]
                node_feature1[fn].insert(2, nf_tmp_list)
                node_feature2[fn].insert(2, nf_tmp_list)
            else:

                node_list[fn][node] = count
                count += 1


        for nd in node_list[fn]:

            if str(nd)[0] == "C":
                for ea in ef_tmp2:
                    if str(ea)[0] == "C":
                        nf_tmp_list = [0, 0, 0, ef_tmp2[ea][1], 0, 0, 0, 0]
                for n in node_list[fn]:
                    if nd == n:
                        index = node_list[fn][n]
                        node_feature1[fn].insert(index, nf_tmp_list)
                        node_feature2[fn].insert(index, nf_tmp_list)
            elif str(nd)[0] == "L":
                for ea in ef_tmp2:
                    if nd == ea:
                        nf_tmp_list = [0, 0, 0, 0, ef_tmp2[ea][2], 0, 0, 0]
                for n in node_list[fn]:
                    if nd == n:
                        index = node_list[fn][n]
                        node_feature1[fn].insert(index, nf_tmp_list)
                        node_feature2[fn].insert(index, nf_tmp_list)
            elif str(nd)[0] == "S":
                if str(nd)[2] == "0":
                    nf_tmp_list1 = [0, 0, 0, 0, 0, 100000, 0, 0]
                    nf_tmp_list2 = [0, 0, 0, 0, 0, 5e-06, 0, 0]
                else:
                    nf_tmp_list1 = [0, 0, 0, 0, 0, 5e-06, 0, 0]
                    nf_tmp_list2 = [0, 0, 0, 0, 0, 100000, 0, 0]

                for n in node_list[fn]:
                    if nd == n:
                        index = node_list[fn][n]
                        node_feature1[fn].insert(index, nf_tmp_list1)
                        node_feature2[fn].insert(index, nf_tmp_list2)
            elif nd != "GND" and nd != "VIN" and nd != "VOUT":
                # print("nd",nd)
                nf_tmp_list = [0, 0, 0, 0, 0, 0, 0, 1]
                for n in node_list[fn]:
                    # print("node list counter",n,type(n))
                    if type(n) == int and nd == n:
                        index = node_list[fn][n]
                        node_feature1[fn].insert(index, nf_tmp_list)
                        node_feature2[fn].insert(index, nf_tmp_list)

        for edge in edge_tmp:
            tmp_list = []
            for nd in node_list[fn]:
                if edge[0] == nd:
                    tmp_list.insert(0,node_list[fn][nd])
                if edge[1] == nd:
                    tmp_list.insert(1, node_list[fn][nd])
            edge_list[fn].append(tmp_list)


        label_feature[fn] = label
        node_feature1[fn] = torch.Tensor(node_feature1[fn])
        node_feature2[fn] = torch.Tensor(node_feature2[fn])
        graph = dgl.graph(edge_list[fn])
        graph = dgl.to_bidirected(graph)
        graph.ndata['node_feature1'] = node_feature1[fn]
        graph.ndata['node_feature2'] = node_feature2[fn]
        graph_list.append(graph)
        label_list.append(label_feature[fn])

    datalist.append(graph_list)
    datalist.append(label_list)

    return datalist




def split_balance_data(dataset, batch_size, rtrain, rval, rtest):
    train_ratio = rtrain
    val_ratio = rval
    test_ratio = rtest
    # print("train_ratio", train_ratio)
    # print("val_ratio", val_ratio)
    # print("test_ratio", test_ratio)

    shuffle_dataset = True
    random_seed = 42

    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    n_train = int(dataset_size * train_ratio)
    n_val = int(dataset_size * val_ratio)
    n_test = int(dataset_size * test_ratio)

    train_indices, val_indices, test_indices = indices[:n_train], indices[n_train + 1:n_train + n_val], indices[
                                                                                                        n_train + n_val + 1:n_train + n_val + n_test]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader


def split_imbalance_data_cls(dataset, batch_size, rtrain, rval, rtest):
    train_ratio = rtrain
    val_ratio = rval
    test_ratio = rtest
    # print("train_ratio", train_ratio)
    # print("val_ratio", val_ratio)
    # print("test_ratio", test_ratio)

    shuffle_dataset = True
    random_seed = 42

    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    n_test = int(dataset_size * test_ratio)

    test_indices = indices[0:n_test]

    ind_positive = []
    ind_negative = []
    ind = 0

    ind_positive = []
    ind_negative = []

    for data in dataset:
        flag_cls = data['label'].tolist()[0]

        if ind in test_indices:
            ind += 1
            continue

        if flag_cls > 0.3:
            ind_positive.append(ind)
        else:
            ind_negative.append(ind)
        ind += 1

    indices_new = ind_negative

    for i in range(int(len(ind_negative) / len(ind_positive))):
        indices_new.extend(ind_positive)

    dataset_size_new = len(indices_new)

    n_train = int(dataset_size_new * train_ratio)
    n_val = int(dataset_size_new * val_ratio)

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices_new)

    train_indices, val_indices, test_indices = indices_new[0:n_train], indices_new[
                                                                       n_train + 1:n_train + n_val], indices_new[
                                                                                                     0:n_train + n_val]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader

def dataset_balance_indices(dataset,indices,reward_th,flag_extend):
    ind = 0
    ind_positive = []
    ind_negative = []

    for data in dataset:
        if ind in indices:
            flag_cls = (data['label_feature'].tolist())[0][0]
            if flag_cls > reward_th:
                ind_positive.append(ind)
            else:
                ind_negative.append(ind)
        ind+=1
    ind_new = ind_negative

    if flag_extend == 1:
        for i in range(int(len(ind_negative)/len(ind_positive))):
            ind_new.extend(ind_positive)
    else:
        ind_new.extend(ind_positive)


    return ind_new




def split_imbalance_data_reward(dataset, batch_size, rtrain, rval, rtest):
    train_ratio = rtrain
    val_ratio = rval
    test_ratio = rtest
    print("train_ratio", train_ratio)
    print("val_ratio", val_ratio)
    print("test_ratio", test_ratio)
    print("dataset", len(dataset))

    shuffle_dataset = True
    random_seed = 42
    reward_th = 0.15


    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    n_train = int(dataset_size * train_ratio)
    n_val = int(dataset_size * val_ratio)
    n_test = int(dataset_size * test_ratio)

    train_indices = indices[0:n_train]
    val_indices = indices[n_train + 1:n_train + n_val]
    test_indices = indices[n_train + n_val + 1:n_train + n_val + n_test]

    train_indices_new = dataset_balance_indices(dataset, train_indices, reward_th, 1)
    print("train indices new", train_indices_new)
    val_indices_new = dataset_balance_indices(dataset, val_indices, reward_th, 1)
    print("val indices new", val_indices_new)
    test_indices_new = dataset_balance_indices(dataset, test_indices, reward_th, 0)
    print("test indices new", test_indices_new)

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(train_indices_new)
        np.random.shuffle(val_indices_new)
        np.random.shuffle(test_indices_new)

    train_sampler = SubsetRandomSampler(train_indices_new)
    valid_sampler = SubsetRandomSampler(val_indices_new)
    test_sampler = SubsetRandomSampler(test_indices_new)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader


