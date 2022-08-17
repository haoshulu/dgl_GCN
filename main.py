import torch
import torch.nn.functional as F
import argparse
import numpy as np
import torch.optim as optim

from utils import train, validation, test, split_data, collect_data
from model import GCN


'''
定义一个显示超参数的函数，将代码中所有的超参数打印
'''
def show_Hyperparameter(args):
    argsDict = args.__dict__
    print(argsDict)
    print('the settings are as following')
    for key in argsDict:
        print(key,':',argsDict[key])

'''
Parameter declare
'''
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode',action='store_true', default=False,
                    help='Validate during traing pass')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate')

parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters)')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability)')

parser.add_argument('-y_select', type=str, default='reg_eff',
                    help='define target label')
parser.add_argument('-train_rate', type=float, default=0.1,
                    help='# components')


# 如果程序不禁止使用gpu且当前主机的gpu可用，arg.cuda就为True
args = parser.parse_args()
show_Hyperparameter(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()

# 指定生成随机数的种子，从而每次生成的随机数都是相同的，通过设定随机数种子的好处是，使模型初始化的可学习参数相同，从而使每次的运行结果可以复现。
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)


'''
开始训练
'''

# 载入数据
dataset = collect_data(args.y_select)
train_loader, validation_loader, test_loader = split_data(args.train_rate, dataset)

# Model and optimizer
model = GCN(in_feats=7,
            h_feats=args.hidden,
            num_classes=1,
            dropout=args.dropout)
print(model)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# 如果可以使用GPU，数据写入cuda，便于后续加速
# .cuda()会分配到显存里（如果gpu可用）
if args.cuda:
    model.cuda()
    dataset = dataset.cuda()


for epoch in range(args.epochs):
    train(model, optimizer, train_loader, epoch)
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        # eval() 函数用来执行一个字符串表达式，并返回表达式的值
        model.eval()
        validation(model, optimizer, validation_loader, epoch)

print("Optimization Finished!")

test(model, test_loader)

torch.cuda.empty_cache()
