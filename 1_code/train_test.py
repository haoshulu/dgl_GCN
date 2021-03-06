import csv
import os
from datetime import datetime
import numpy as np
import torch
from torch.nn import Linear, MSELoss

from topo_data import *
from ml_utils import train, test
from model_dgl import *
import argparse


if __name__ == '__main__':

# ======================== Arguments ==========================#

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', type=str, default="../0_rawdata", help='raw data path')
    parser.add_argument('-y_select', type=str, default='reg_reward', help='define target label')
    parser.add_argument('-batch_size', type=int, default=256, help='batch size')
    parser.add_argument('-n_epoch', type=int, default=100, help='number of training epoch')
    parser.add_argument('-gnn_nodes', type=int, default=20, help='number of nodes in hidden layer in GNN')
    parser.add_argument('-predictor_nodes', type=int, default=10, help='number of MLP predictor nodes at output of GNN')
    parser.add_argument('-gnn_layers', type=int, default=2, help='number of layer')
    parser.add_argument('-model_index', type=int, default=4, help='model index')
    parser.add_argument('-threshold', type=float, default=0, help='classification threshold')
    parser.add_argument('-ncomp', type=int, default=5, help='# components')
    parser.add_argument('-train_rate', type=float, default=0.6, help='# components')


    parser.add_argument('-retrain', type=int, default=1, help='force retrain model')
#    parser.add_argument('-seed', type=int, default=0, help='random seed')
    parser.add_argument('-seedrange', type=int, default=1, help='random seed')
    parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-weight_decay', type=float, default=5e-4, help='weight decay')

    #parser.add_argument('-loops_num',type=int, default=6,help='loops number for edge attr encoder')
#----------------------------------------------------------------------------------------------------#

    args = parser.parse_args()
    print("\nargs: ",args)


    ncomp=args.ncomp
    train_rate=args.train_rate
    path=args.path
    y_select=args.y_select
    gnn_layers=args.gnn_layers
    gnn_nodes=args.gnn_nodes
    data_folder='../2_dataset/'+y_select+'_'+str(ncomp)
    batch_size=args.batch_size
    n_epoch=args.n_epoch
    th=args.threshold
    model_index=args.model_index
    retrain=args.retrain

    lr = args.lr
    weight_decay = args.weight_decay
    seedrange = args.seedrange

    output_file = datetime.now().strftime(y_select + '-'  + str(ncomp))
    final_result = []


# ======================== Data & Model ==========================#

    dataset = collectData(path,y_select,ncomp)

    # print("dataset", dataset)
    # print('\n # data point:\n', len(dataset))



    # # set random seed for training
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    for seed in range(seedrange):
        # set random seed for training
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print("seed: ", seed)

        nf_size=4
        ef_size=3
        nnode=8
        # if args.model_index==0:
        #     ef_size=6

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset_size = len(dataset[0])
        indices = list(range(dataset_size))
        shuffle_dataset = True
        random_seed = 42
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        n_test = int(dataset_size * (0.9-train_rate))
        n_train = int(dataset_size * train_rate)
        n_val = int(dataset_size * 0.1)

        train_loader = [dataset[0][0:n_train], dataset[1][0:n_train]]
        val_loader = [dataset[0][n_train+1:n_train+n_val], dataset[1][n_train+1:n_train+n_val]]
        test_loader = [dataset[0][n_train+n_val+1:-1], dataset[1][n_train+n_val+1:-1]]

        model = GCN(8, 16, 1).to(device)



        print('training')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = MSELoss(reduction='mean').to(device)
        train(train_loader=train_loader,
                           val_loader=val_loader,
                           model=model,
                           n_epoch=n_epoch,
                           batch_size=batch_size,
                           device=device,
                           optimizer=optimizer)



        accuracy = test(test_loader=test_loader, model=model,device=device)
        print("accuracy", accuracy)

        # final_result.append([model_index, ncomp, y_select, gnn_layers, gnn_nodes,
        #                      min_loss, mean_loss, final_rse, rse_bins[0], rse_bins[1], rse_bins[2]])

# with open('./log/result-'+output_file + '.csv','w') as f:
#         csv_writer = csv.writer(f)
#         header = ['model_index','n_comp','y_select','gnn_layers','gnn_nodes',
#                   'min_loss','mean_loss','final_rse','mse[0-0.3]','mse[0.3-0.7]','mse[0.7-1]']
#         csv_writer.writerow(header)
#
#         csv_writer.writerows(final_result)




