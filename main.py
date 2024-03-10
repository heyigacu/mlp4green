import os
import pandas as pd
from data_clean import data_preprocess, statistics
from load_data import *
from feature import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter
from model import MLP, MPNN, GraphSAGEPredictor
from trainer import train_mlp, train_ml, train_gnn
import numpy as np

parent_dir = os.path.abspath(os.path.dirname(__file__))
print(parent_dir)


data_path = parent_dir+"/dataset/train/cleaned_data_sampled.csv"

df = pd.read_csv(data_path,sep='\t',header=0)
smileses = df['Smiles']
labels = df['Label'].astype(int)
print('read data:', Counter(labels))
graphs = [Graph_smiles(smiles) for smiles in smileses]
nBits = 2048
features = np.array([morgan_featurizer(smiles, nBits=nBits) for smiles in smileses])

def mlp_bi_class():
    tuple_ls = list(zip(features, labels))
    #########
    # mlp
    #########
    batchsize = int(len(tuple_ls)/16)
    drop_last = True
    n_feats = 2048
    n_hiddens = 256
    n_tasks = 2

    model = MLP(n_feats=n_feats, n_hiddens=n_hiddens, n_tasks=n_tasks)
    kfolds = BaseDataLoader.load_data_kfold_torch_batchsize(tuple_ls, batchsize=batchsize, Stratify=True, drop_last=drop_last, sample_method=None)
    print('mlp start training!')
    rst_mlp, best_epochs = train_mlp.train_bi_classify_kfolds(model, kfolds=kfolds, max_epochs=500, patience=7, save_folder=parent_dir+'/pretrained/',save_name='kfolds.pth')
    print('optimization finished!', rst_mlp)
    return rst_mlp, best_epochs

def svm_bi_class():
    tuple_ls = list(zip(features, labels))
    #########
    # svm
    #########
    from sklearn import svm
    model =svm.SVC(probability=True)    
    kfolds = BaseDataLoader.load_data_kfold_numpy(tuple_ls, Stratify=True)
    print('svm start training!')
    rst_svm = train_ml.train_bi_classify_kfolds(model=model, kfolds=kfolds)
    print('optimization finished!', rst_svm)
    return rst_svm


def rf_bi_class():
    tuple_ls = list(zip(features, labels))
    #########
    # rf
    #########
    from sklearn.ensemble import RandomForestClassifier
    kfolds = BaseDataLoader.load_data_kfold_numpy(tuple_ls, Stratify=True)
    model = RandomForestClassifier()
    print('rf start training!')
    rst_rf = train_ml.train_bi_classify_kfolds(model=model, kfolds=kfolds)
    print('optimization finished!', rst_rf)
    return rst_rf


def gnn_bi_class():
    tuple_ls = list(zip(graphs, labels))
    #########
    # svm
    #########
    n_feats = 74
    edge_in_feats = 12
    n_tasks = 2
    node_out_feats = 256
    edge_hidden_feats = 256
    batchsize = int(len(tuple_ls)/16)

    model = MPNN(node_in_feats=n_feats, edge_in_feats=edge_in_feats, node_out_feats=node_out_feats, edge_hidden_feats=edge_hidden_feats, n_tasks=n_tasks)
    kfolds = BaseDataLoader.load_data_kfold_graph_batchsize(tuple_ls, batchsize, Stratify=True, drop_last=True)
    print('mpnn start training!')
    rst_gnn = train_gnn.train_bi_classify_kfolds(model, kfolds=kfolds, edge=True, max_epochs=500, patience=7, save_folder=parent_dir+'/pretrained/',save_name='gnn.pth')
    print('optimization finished!', rst_gnn)

    return rst_gnn


def total():
    total_rst = []
    rst_mlp, best_epochs = mlp_bi_class()
    rst_svm = svm_bi_class()
    rst_rf = rf_bi_class()
    rst_gnn = gnn_bi_class()
    total_rst += [rst_mlp, rst_rf, rst_svm, rst_gnn]
    np.savetxt(parent_dir+"/analysis/model_result/performance.txt", np.array(total_rst))


def train():
    tuple_ls = list(zip(features, labels))
    batchsize = int(len(tuple_ls)/16)
    drop_last = True
    n_feats = 2048
    n_hiddens = 256
    n_tasks = 2

    rst_mlp, best_epochs = mlp_bi_class()
    model = MLP(n_feats=n_feats, n_hiddens=n_hiddens, n_tasks=n_tasks)
    all = BaseDataLoader.load_data_all_torch_batchsize(tuple_ls, batchsize, drop_last=True)
    train_mlp.train_bi_classify_all(model, all=all, best_epochs=best_epochs, patience=7, save_folder=parent_dir+'/pretrained/',save_name='all_mlp.pth')
    
total()
train()
