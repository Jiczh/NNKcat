#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import r2_score
import torch.optim as optim
import torch.utils.data as Data
torch.manual_seed(8) # for reproduce

import time
import numpy as np
import gc
import sys
sys.setrecursionlimit(50000)
import pickle
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')
# from tensorboardX import SummaryWriter
torch.nn.Module.dump_patches = True
import copy
import pandas as pd
#then import my own modules
from AttentiveFP import Fingerprint, Fingerprint_viz, save_smiles_dicts, get_smiles_dicts, get_smiles_array, moltosvg_highlight


# In[2]:


from rdkit import Chem
# from rdkit.Chem import AllChem
from rdkit.Chem import QED
from rdkit.Chem import rdMolDescriptors, MolSurf
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
#get_ipython().run_line_magic('matplotlib', 'inline')
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.cm as cm
import matplotlib
import seaborn as sns; sns.set_style("darkgrid")
from IPython.display import SVG, display
import sascorer
import itertools
#from sklearn.metrics import r2_score
import scipy


# In[3]:


random_seed = 0 # 69, 88
start_time = str(time.ctime()).replace(':','-').replace(' ','_')

batch_size = 100
epochs = 200

p_dropout= 0.2
fingerprint_dim = 200

weight_decay = 5 # also known as l2_regularization_lambda
learning_rate = 2.5
output_units_num = 1 # for regression model
radius = 2
T = 4


# In[4]:


task_name = 'kcat'
tasks = ['ln2']

#raw_filename = "../data/uniq_train_liglist.csv"
#raw_filename = "../data/train_1357.csv"
raw_filename = "../yeast/yeast_input_mol.csv"
feature_filename = raw_filename.replace('.csv','.pickle')
filename = raw_filename.replace('.csv','')
prefix_filename = raw_filename.split('/')[-1].replace('.csv','')
#smiles_tasks_df = pd.read_csv(raw_filename,sep=',')
#smiles_tasks_df=smiles_tasks_df.rename(columns={0:'ID',1:'density',2:'smiles'})
new_column_names = ['Smiles', 'value']
smiles_tasks_df = pd.read_csv(raw_filename,sep=',', header=0, names=new_column_names)
smilesList = smiles_tasks_df.Smiles.values
print("number of all smiles: ",len(smilesList))
atom_num_dist = []
remained_smiles = []
canonical_smiles_list = []
for smiles in smilesList:
    try:        
        mol = Chem.MolFromSmiles(smiles)
        atom_num_dist.append(len(mol.GetAtoms()))
        remained_smiles.append(smiles)
        canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
    except:
        print(smiles)
        pass
print("number of successfully processed smiles: ", len(remained_smiles))
smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["Smiles"].isin(remained_smiles)]
# print(smiles_tasks_df)
smiles_tasks_df['cano_smiles'] =canonical_smiles_list
#assert canonical_smiles_list[8]==Chem.MolToSmiles(Chem.MolFromSmiles(smiles_tasks_df['cano_smiles'][8]), isomericSmiles=True)

plt.figure(figsize=(5, 3))
sns.set(font_scale=1.5)
ax = sns.distplot(atom_num_dist, bins=28, kde=False)
plt.tight_layout()
# plt.savefig("atom_num_dist_"+prefix_filename+".png",dpi=200)
plt.show()
plt.close()


# In[5]:


smiles_tasks_df


# In[6]:


smilesList_rest = smiles_tasks_df.Smiles.values


# In[7]:


if os.path.isfile(feature_filename):
    feature_dicts = pickle.load(open(feature_filename, "rb" ))
else:
    feature_dicts = save_smiles_dicts(smilesList_rest,filename)
# feature_dicts = get_smiles_dicts(smilesList)
remained_df = smiles_tasks_df[smiles_tasks_df["cano_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
uncovered_df = smiles_tasks_df.drop(remained_df.index)
print("not processed items")
uncovered_df


# In[8]:


def train(model, dataset, optimizer, loss_function):
    model.train()
    np.random.seed(epoch)
    valList = np.arange(0,dataset.shape[0])
    #shuffle them
    np.random.shuffle(valList)
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch)   
    for counter, train_batch in enumerate(batch_list):
        batch_df = dataset.loc[train_batch,:]
        smiles_list = batch_df.cano_smiles.values
        y_val = batch_df[tasks[0]].values
        
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,feature_dicts)
        atoms_prediction, mol_prediction = model(torch.Tensor(x_atom),torch.Tensor(x_bonds),torch.cuda.LongTensor(x_atom_index),torch.cuda.LongTensor(x_bond_index),torch.Tensor(x_mask))
        
        model.zero_grad()
        loss = loss_function(mol_prediction, torch.Tensor(y_val).view(-1,1))     
        loss.backward()
        optimizer.step()
def eval(model, dataset):
    model.eval()
    eval_MAE_list = []
    eval_MSE_list = []
    valList = np.arange(0,dataset.shape[0])
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch) 
    for counter, eval_batch in enumerate(batch_list):
        batch_df = dataset.loc[eval_batch,:]
        smiles_list = batch_df.cano_smiles.values
        y_val = batch_df[tasks[0]].values
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,feature_dicts)
        atoms_prediction, mol_prediction = model(torch.Tensor(x_atom),torch.Tensor(x_bonds),torch.cuda.LongTensor(x_atom_index),torch.cuda.LongTensor(x_bond_index),torch.Tensor(x_mask))
        #print( torch.Tensor(y_val).view(-1,1))
        MAE = F.l1_loss(mol_prediction, torch.Tensor(y_val).view(-1,1), reduction='none')        
        MSE = F.mse_loss(mol_prediction, torch.Tensor(y_val).view(-1,1), reduction='none')
        #r2 = r2.append(r2_score(mol_prediction, torch.Tensor(y_val).view(-1,1)))
        #r2 = r2_score(mol_prediction, torch.Tensor(y_val).view(-1,1))
        eval_MAE_list.extend(MAE.data.squeeze().cpu().numpy())
        eval_MSE_list.extend(MSE.data.squeeze().cpu().numpy())
        #eval_r2_list.append(r2)
        #print(eval_r2_list)
    return np.array(eval_MAE_list).mean(), np.array(eval_MSE_list).mean()


# In[9]:


torch.cuda.empty_cache()


# In[10]:


# evaluate model
best_model = torch.load('saved_models/kcat_model_uniq_train_liglist_Mon_May__6_15-00-41_2024_10.pt')  

# In[11]:


best_model


# In[12]:


best_model.output = torch.nn.Sequential(torch.nn.Linear(in_features=100, out_features=30, bias=True))     ################################################output 


# In[13]:


smiles_list=remained_df.cano_smiles.values


# In[14]:


#remained_df=remained_df.set_index('Entry')


# In[15]:


remained_df


# In[16]:

start_row = int(sys.argv[1])
end_row = int(sys.argv[2])
sub = int(sys.argv[3])

remained_df1=remained_df.iloc[start_row:end_row,:]




smiles_list1=remained_df1.cano_smiles.values
x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list1,feature_dicts)
atoms_prediction, mol_prediction = best_model(torch.Tensor(x_atom),torch.Tensor(x_bonds),torch.cuda.LongTensor(x_atom_index),torch.cuda.LongTensor(x_bond_index),torch.Tensor(x_mask))
print(mol_prediction.shape)
df_features=pd.DataFrame(data=mol_prediction.cpu().data.numpy(),index=remained_df1.index)


# In[20]:


#df_features.to_csv('../test/test_30features_{}.csv'.format(sub),header=None)
#df_features.to_csv('../test/train_30features_{}.csv'.format(sub),header=None)
df_features.to_csv('../yeast/yeast_30features_{}.csv'.format(sub),header=None)





