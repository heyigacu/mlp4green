import argparse
import torch
import os
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import pandas as pd
from model import MLP
from feature import morgan_featurizer

work_dir = os.path.abspath(os.path.dirname(__file__))

parser = argparse.ArgumentParser(description='molecules odor predictor')
parser.add_argument("-i", "--input", type=str, default=work_dir+'/example.smi', help="input file")
parser.add_argument("-o", "--output", type=str, default=work_dir+'/result.csv',help="output file")
args = parser.parse_args()

##################
# predict
##################

smileses = list(pd.read_csv(args.input, header=0, sep=' ')['Smiles'])

def mlp_bi(model, path, smileses):
    state_dict = torch.load(work_dir+"/"+path)
    model.load_state_dict(state_dict)
    model.eval()
    total = []
    for i,smiles in enumerate(smileses):
        try:
            if i % 10000 ==0:
                print(i)
            for i in list(DataLoader([morgan_featurizer(smiles)], batch_size=1, shuffle=False, collate_fn=None, drop_last=False)):
                feature = i
            rst = model(feature)
            rst =  F.softmax(rst,dim=1).detach().numpy()[0]
            labels = ['non-green', 'green']
            string = labels[rst.argmax()]
            ls = []
            ls.append(string)
            for value in rst:
                ls.append('{:.4f}'.format(value))
            total.append(ls)
        except:
            total.append(['error smiles', 'nan', 'nan'])
    return total

def mlp_multi(model, path, smileses):
    state_dict = torch.load(work_dir+"/"+path)
    model.load_state_dict(state_dict)
    model.eval()
    total = []
    for i,smiles in enumerate(smileses):
        try:
            if i % 10000 ==0:
                print(i)
            for i in list(DataLoader([morgan_featurizer(smiles)], batch_size=1, shuffle=False, collate_fn=None, drop_last=False)):
                feature = i
            rst = model(feature)
            rst =  F.softmax(rst,dim=1).detach().numpy()[0]
            labels = ['non-green', 'green']
            string = labels[rst.argmax()]
            ls = []
            ls.append(string)
            for value in rst:
                ls.append('{:.4f}'.format(value))
            total.append(ls)
        except:
            total.append(['error smiles', 'nan', 'nan'])
    return total

model = MLP(n_feats = 2048, n_hiddens = 256, n_tasks = 2)
path = "/pretrained/all_mlp.pth"
total = mlp_bi(model, path, smileses)
print(total)
df = pd.DataFrame(total)
df.columns = ['Odor', 'non-green', 'green']
df.insert(0,'Smiles',smileses)
df.to_csv(args.output,index=False,header=True,sep='\t')
