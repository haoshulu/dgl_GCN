import pprint

import sklearn.metrics
import torch
import torch.nn.functional as F

import numpy as np
import math
from sklearn.metrics import roc_auc_score,roc_curve,auc, explained_variance_score
from sklearn.utils.multiclass import type_of_target
from easydict import EasyDict

from model_dgl import  *
import copy


def rse(y,yt):

    assert(y.shape==yt.shape)

    if len(y)==0:
        return 0,0
    var=0
    m_yt=yt.mean()
#    print(yt,m_yt)
    for i in range(len(yt)):
        var+=(yt[i]-m_yt)**2
    print("len(y)",len(y))
    var = var/len(y)
    mse=0
    for i in range(len(yt)):
        mse+=(y[i]-yt[i])**2
    mse = mse/len(y)
    # print("var: ", var)
    # print("mse: ",mse)
    rse=mse/(var+0.0000001)

    rmse=math.sqrt(mse/len(yt))

#    print(rmse)

    return rse,mse

def compute_roc(preds, ground_truth):
    """
    Generate TPR, FPR points under different thresholds for the ROC curve.
    Reference: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
    :param preds: a list of surrogate model predictions
    :param ground_truth: a list of ground-truth values (e.g. by the simulator)
    :return: {threshold: {'TPR': true positive rate, 'FPR': false positive rate}}
    """
    preds, ground_truth = np.array(preds), np.array(ground_truth)
    th_true = 0.6

    for i in range(len(ground_truth)):

        if ground_truth[i] <= th_true:
            ground_truth[i] = 0
        else:
            ground_truth[i] = 1

    #thresholds = np.arange(0.1, 1., step=0.1)  # (0.1, 0.2, ..., 0.9)
    result = {}
    print("ground truth", ground_truth)
    print("prediction",preds)

    fpr, tpr, threshold = roc_curve(ground_truth, preds,pos_label=1)
    AUC = auc(fpr, tpr)
    # for thres in thresholds:
    #     gt_pos = np.where(ground_truth >= thres)[0]
    #     gt_neg = np.where(ground_truth < thres)[0]
    #     print("gt_pos", gt_pos, len(gt_pos))
    #     print("ge_neg",gt_neg, len(gt_neg))
    #     predict_pos = np.where(preds >= thres)[0]
    #     print("predict_pos",predict_pos)
    #
    #     if len(gt_pos) == 0:
    #         TPR = 0
    #     else:
    #         TPR = len(np.intersect1d(gt_pos, predict_pos)) / len(gt_pos)
    #     print("TPR",TPR)
    #
    #     if len(gt_neg) == 0:
    #         FPR = 0
    #     else:
    #         FPR = len(np.intersect1d(gt_neg, predict_pos)) / len(gt_neg)
    #
    #     result[thres] = {'TPR': TPR, 'FPR': FPR}

    # preds, ground_truth = np.array(preds), np.array(ground_truth)
    # print("preds", preds,len(preds))
    # print("ground truth", ground_truth,len(ground_truth))
    # score = explained_variance_score(ground_truth, preds)
    # print("score", score)
    # FPR, TPR, threshold = roc_curve(ground_truth, preds)
    # AUC = auc(FPR, TPR)
    # result = {}
    # for th in threshold:
    #     result[th] = {'TPR': TPR, 'FPR': FPR}
    # print("result", result)
    # print("AUC", auc)

    print("fpr, tpr, threshold",fpr, tpr, threshold)
    return 0



def train(train_loader, val_loader, model, n_epoch, batch_size, device, optimizer):
    train_perform = []
    min_val_loss = 100
    graphlist = train_loader[0]
    labellist = train_loader[1]
    train_loader = zip(graphlist, labellist)
    graphlist_val = val_loader[0]
    labellist_val = val_loader[1]
    val_loader = zip(graphlist_val, labellist_val)

    for epoch in range(n_epoch):

        ########### Training #################

        train_loss = 0
        n_batch_train = 0
        model.train()
        for i, data in enumerate(train_loader):
            #data.to(device)
            #print("data in train loader", data)
            graph = data[0]
            #print("graph", graph)
            y = torch.tensor(data[1])
            #print("label", y)
            y = y.to(torch.float32)
            if torch.cuda.is_available():
                graph = graph.to('cuda')
                y= y.to('cuda')


            n_batch_train = n_batch_train + 1

            out = model(graph, graph.ndata['node_feature1'].float(), graph.ndata['node_feature2'].float())
            out = out.reshape(y.shape)
            assert (out.shape == y.shape)
            loss = F.mse_loss(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += out.shape[0] * loss.item()

        if epoch % 1 == 0:
            # print("n batch trian", n_batch_train)
            # print("batch size", batch_size)
            # print('%d epoch training loss: %.3f' % (epoch, train_loss / n_batch_train / batch_size))

            n_batch_val = 0
            val_loss = 0

            #                epoch_min=0
            model.eval()

            for i, data in enumerate(val_loader):
                #print("data in val loader", data)
                n_batch_val += 1
#                data.to(device)
                graph = data[0]
                # print("graph", graph)
                y = torch.tensor(data[1])
                # print("label", y)
                y = y.to(torch.float32)
                if torch.cuda.is_available():
                    graph = graph.to('cuda')
                    y = y.to('cuda')
                n_batch_train = n_batch_train + 1
                optimizer.zero_grad()

                out = model(graph, graph.ndata['node_feature1'].float(), graph.ndata['node_feature2'].float())

                out = out.reshape(y.shape)
                assert (out.shape == y.shape)
                #                     loss=F.binary_cross_entropy(out, y.float())
                loss = F.mse_loss(out, y.float())
                val_loss += out.shape[0] * loss.item()

            # print(n_batch_val)
            # print("batch train", n_batch_train)
            # val_loss_ave = val_loss / n_batch_val / batch_size

    #         if val_loss_ave < min_val_loss:
    #             model_copy = copy.deepcopy(model)
    #             print('lowest val loss', val_loss_ave)
    #             epoch_min = epoch
    #             min_val_loss = val_loss_ave
    #
    #         if epoch - epoch_min > 5:
    #             # print("training loss:",train_perform)
    #             print("training loss minimum value:", min(train_perform))
    #             print("training loss average value:", np.mean(train_perform))
    #
    #             return model_copy, min(train_perform), np.mean(train_perform)
    #
    #     train_perform.append(train_loss / n_batch_train / batch_size)
    #
    # return model, min(train_perform), np.mean(train_perform)


def test(test_loader, model, device):
    model.eval()
    accuracy = 0
    n_batch_test = 0
    gold_list = []
    out_list = []
    analytic_list = []
    graphlist = test_loader[0]
    labellist = test_loader[1]
    test_loader = zip(graphlist, labellist)

    num_correct = 0
    num_tests = 0

    for i, data in enumerate(test_loader):
        #data.to(device)
        #print("data in test loader",data)
        graph = data[0]
        #print("graph", graph)
        y = torch.tensor(data[1])
        #print("label", y)
        y = y.to(torch.float32)
        if torch.cuda.is_available():
            graph = graph.to('cuda')
            y = y.to('cuda')

        n_batch_test = n_batch_test + 1
        out = model(graph, graph.ndata['node_feature1'].float(), graph.ndata['node_feature2'].float())

        num_correct += (out.argmax(1) == y).sum().item()
        num_tests += len(y)

    return num_correct/num_tests




def compute_errors_by_bins(pred_y:np.array, true_y:np.array, bins):
    """
    Divide data by true_y into bins, report their rse separately
    :param pred_y: model predictions (of the test data)
    :param true: true labels (of the test data)
    :param bins: a list of ranges where errors in these ranges are computed separately
                 e.g. [(0, 0.33), (0.33, 0.66), (0.66, 1)]
    :return: a list of rses by bins
    """
    results = []

    for range_from, range_to in bins:
        # get indices of data in this range
        indices = np.nonzero(np.logical_and(range_from <= true_y, true_y < range_to))

        if len(indices) > 0:
            temp_rse, temp_mse = rse(pred_y[indices], true_y[indices])
            results.append(math.sqrt(temp_mse))
            # print('data between ' + str(range_from) + ' ' + str(range_to))
            # pprint.pprint(list(zip(pred_y[indices], true_y[indices])))
        else:
            print('empty bin in the range of ' + str(range_from) + ' ' + str(range_to))

    return results
