import json

import torch.nn.functional as F
import dgl
import torch
import numpy as np


def train(model, optimizer, train_loader, epoch):
    graphlist = train_loader[0]
    labellist = train_loader[1]
    train_loader = zip(graphlist, labellist)
    train_loss = 0
    for n in range(epoch):
        for i, data in enumerate(train_loader):
            graph = data[0]
            y = torch.tensor(data[1])
            model.train()

            # graph = train_loader[0]
            # y = torch.tensor(train_loader[1]).to(torch.float32)
            # optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
            # pytorch中每一轮batch需要设置optimizer.zero_grad
            optimizer.zero_grad()

            # 计算输出时，对所有的节点都进行计算
            output = model(graph, graph.ndata['node_feature1'], graph.ndata['node_feature2'])
            # 损失函数，仅对训练集的节点进行计算，即：优化对训练数据集进行
            loss = F.mse_loss(output.reshape(y.shape), y)
            # 计算准确率
            #acc_train = accuracy(output, y)
            # 反向求导  Back Propagation
            loss.backward()
            # 更新所有的参数
            optimizer.step()
            # 通过计算训练集损失和反向传播及优化，带标签的label信息就可以smooth到整个图上（label information is smoothed over the graph）。

            train_loss +=  loss.item()


        print('Epoch: {:04d}'.format(n + 1),
              'loss_train: {:.4f}'.format(train_loss),
              )


def validation(model, optimizer, val_loader, epoch):
    graphlist = val_loader[0]
    labellist = val_loader[1]
    val_loader = zip(graphlist, labellist)

    val_loss = 0

    for n in range(epoch):
        for i, data in enumerate(val_loader):
            graph = data[0]
            y = torch.tensor(data[1]).to(torch.float32)

            model.eval()
            optimizer.zero_grad()
            output = model(graph, graph.ndata['node_feature1'], graph.ndata['node_feature2'])
            # 验证集的损失函数
            loss = F.mse_loss(output.reshape(y.shape), y)
            # acc_val = accuracy(output, y)

            val_loss += loss.item()

        print('Epoch: {:04d}'.format(n + 1),
              'loss_val: {:.4f}'.format(val_loss),
              )


def test(model, test_loader):
    graphlist = test_loader[0]
    labellist = test_loader[1]
    test_loader = zip(graphlist, labellist)
    test_loss = 0
    num_correct = 0

    for i, data in enumerate(test_loader):

        model.eval()
        graph = data[0]
        y = torch.tensor(data[1]).to(torch.float32)
        output = model(graph, graph.ndata['node_feature1'], graph.ndata['node_feature2'])
        loss = F.mse_loss(output.reshape(y.shape), y)
        test_loss += loss.item()
        num_correct += (output == y).sum().item()

    print("Test set results:",
          "loss= {:.4f}".format(test_loss),
          'accuracy={:.4f}'.format(num_correct/len(labellist)))



def load_data(y_select):
    '''
    从json文件中获取 tmp
    '''
    json_file = json.load(open("../0_data/dataset_3.json"))

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


        target_eff.append(eff)
        target_vout.append(vout)


        if y_select == 'reg_eff':
            label = target_eff

        elif y_select == 'reg_vout':
            label = target_vout

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

        if param_id == '0':
            reward_max = -1
        if param_id == '2':
            tmp[topo_id] = {}
            tmp[topo_id]['nodes'] = list_of_node
            tmp[topo_id]['edges'] = list_of_edge
            tmp[topo_id]['edge_attr1'] = edge_attr0
            tmp[topo_id]['edge_attr2'] = edge_attr
            tmp[topo_id]['node_attr'] = node_attr
            tmp[topo_id]['rout'] = rout
            tmp[topo_id]['cout'] = cout
            tmp[topo_id]['label'] = label

        if param_id == '4' and topo_id in tmp.keys():
            # if reward_max<0.1, delete topo

            #                print(label,nn)
            nn = nn + 1

            tmp[topo_id]['sim_eff'] = target_eff
            tmp[topo_id]['sim_vout'] = target_vout
            tmp[topo_id]['analytic_eff'] = label_analytic_eff
            tmp[topo_id]['analytic_vout'] = label_analytic_vout
            reward_max = -1

        else:

            continue
    return tmp

def collect_data(y_select):
    '''
    将tmp转化为可用的datalist
    :return: datalist
    '''
    tmp = load_data(y_select)
    node_list = {}
    edge_list = {}
    node_feature1 = {}
    node_feature2 = {}
    label_feature = {}
    graph_list = []
    label_list = []
    datalist = []

    key_list = list(tmp.keys())
    np.random.shuffle(key_list)
    for fn in key_list:
        nodes_tmp = tmp[fn]['nodes']
        edge_tmp = tmp[fn]['edges']
        ef_tmp1 = tmp[fn]['edge_attr1']
        ef_tmp2 = tmp[fn]['edge_attr2']
        label = tmp[fn]['label']

        node_list[fn] = {}
        edge_list[fn] = []
        node_feature1[fn] = []
        node_feature2[fn] = []
        nf_tmp_list = []
        count = 3
        for node in nodes_tmp:

            if node == "VIN":
                node_list[fn]["VIN"] = 0
                nf_tmp_list = [1, 0, 0, 0, 0, 0, 0]
                node_feature1[fn].insert(0, nf_tmp_list)
                node_feature2[fn].insert(0, nf_tmp_list)
            elif node == "VOUT":
                node_list[fn]["VOUT"] = 1
                nf_tmp_list = [0, 1, 0, 0, 0, 0, 0]
                node_feature1[fn].insert(1, nf_tmp_list)
                node_feature2[fn].insert(1, nf_tmp_list)
            elif node == "GND":
                node_list[fn]["GND"] = 2
                nf_tmp_list = [0, 0, 1, 0, 0, 0, 0]
                node_feature1[fn].insert(2, nf_tmp_list)
                node_feature2[fn].insert(2, nf_tmp_list)
            else:

                node_list[fn][node] = count
                count += 1

        for nd in node_list[fn]:

            if str(nd)[0] == "C":
                for ea in ef_tmp2:
                    if str(ea)[0] == "C":
                        nf_tmp_list = [0, 0, 0, ef_tmp2[ea][1], 0, 0, 0]
                for n in node_list[fn]:
                    if nd == n:
                        index = node_list[fn][n]
                        node_feature1[fn].insert(index, nf_tmp_list)
                        node_feature2[fn].insert(index, nf_tmp_list)
            elif str(nd)[0] == "L":
                for ea in ef_tmp2:
                    if nd == ea:
                        nf_tmp_list = [0, 0, 0, 0, ef_tmp2[ea][2], 0, 0]
                for n in node_list[fn]:
                    if nd == n:
                        index = node_list[fn][n]
                        node_feature1[fn].insert(index, nf_tmp_list)
                        node_feature2[fn].insert(index, nf_tmp_list)
            elif str(nd)[0] == "S":
                if str(nd)[2] == "0":
                    nf_tmp_list1 = [0, 0, 0, 0, 0, 100000, 0]
                    nf_tmp_list2 = [0, 0, 0, 0, 0, 5e-06, 0]
                else:
                    nf_tmp_list1 = [0, 0, 0, 0, 0, 5e-06, 0]
                    nf_tmp_list2 = [0, 0, 0, 0, 0, 100000, 0]

                for n in node_list[fn]:
                    if nd == n:
                        index = node_list[fn][n]
                        node_feature1[fn].insert(index, nf_tmp_list1)
                        node_feature2[fn].insert(index, nf_tmp_list2)
            elif nd != "GND" and nd != "VIN" and nd != "VOUT":
                nf_tmp_list = [0, 0, 0, 0, 0, 0, 1]
                for n in node_list[fn]:
                    if type(n) == int and nd == n:
                        index = node_list[fn][n]
                        node_feature1[fn].insert(index, nf_tmp_list)
                        node_feature2[fn].insert(index, nf_tmp_list)

        for edge in edge_tmp:
            tmp_list = []
            for nd in node_list[fn]:
                if edge[0] == nd:
                    tmp_list.insert(0, node_list[fn][nd])
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


def split_data(train_rate, dataset):
    dataset_size = len(dataset[0])
    n_train = int(dataset_size * train_rate)
    n_val = int(dataset_size * 0.1)

    train_loader = [dataset[0][0:n_train], dataset[1][0:n_train]]
    val_loader = [dataset[0][n_train + 1:n_train + n_val], dataset[1][n_train + 1:n_train + n_val]]
    test_loader = [dataset[0][n_train + n_val + 1:-1], dataset[1][n_train + n_val + 1:-1]]

    return train_loader, val_loader, test_loader