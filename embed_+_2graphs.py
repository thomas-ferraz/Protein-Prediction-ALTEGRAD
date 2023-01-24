

import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GINConv,GINEConv,PDNConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils.convert import from_scipy_sparse_matrix

import csv
import time
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm

import torch.nn as nn
from torch import optim
import os

import csv
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression

# Read sequences
sequences = list()
with open('sequences.txt', 'r') as f:
    for line in f:
        sequences.append(line[:-1])

# Split data into training and test sets
sequences_train = list()
sequences_test = list()
proteins_test = list()
y_train = list()
with open('graph_labels.txt', 'r') as f:
    for i,line in enumerate(f):
        t = line.split(',')
        if len(t[1][:-1]) == 0:
            proteins_test.append(t[0])
            sequences_test.append(sequences[i])
        else:
            sequences_train.append(sequences[i])
            y_train.append(int(t[1][:-1]))

# Map sequences to 
#vec = CountVectorizer(analyzer='char', ngram_range=(1, 6))
#X_train_sequence_ngram = vec.fit_transform(sequences_train)
#X_test_sequence_ngram = vec.transform(sequences_test)

X_train_sequence_ngram.shape

import pickle
with open('X_train_sequence_ProtTransBertBFDEmbedder.pkl', 'rb') as f:
    X_train_sequence = np.vstack(pickle.load(f))

X_train_sequence[0]

with open('X_test_sequence_ProtTransBertBFDEmbedder.pkl', 'rb') as f:
    X_test_sequence = np.vstack(pickle.load(f))

# Normalize vectors
for i in range(X_train_sequence.shape[0]):
  X_train_sequence[i] = X_train_sequence[i]/len(sequences_train[i])
for i in range(X_test_sequence.shape[0]):
  X_test_sequence[i] = X_test_sequence[i]/len(sequences_test[i])

X_train_sequence[0]

X_train_sequence.shape

sizes = []
for elem in sequences_train:
  sizes.append(len(elem))
sizes = np.array(sizes)

print(np.mean(sizes))
print(np.max(sizes))

np.std(sizes)

sequences[1]

input_seq_dim = X_train_sequence.shape[1]

torch.cuda.empty_cache()

import gc
gc.collect()

torch.cuda.empty_cache()

import gc
gc.collect()

from numba import cuda 
device = cuda.get_current_device()
device.reset()

!nvidia-smi

import csv
import time
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

def load_data(): 
    """
    Function that loads graphs
    """  
    graph_indicator = np.loadtxt("graph_indicator.txt", dtype=np.int64)
    _,graph_size = np.unique(graph_indicator, return_counts=True)
    
    edges = np.loadtxt("edgelist.txt", dtype=np.int64, delimiter=",")
    edges_inv = np.vstack((edges[:,1], edges[:,0]))
    edges = np.vstack((edges, edges_inv.T))
    s = edges[:,0]*graph_indicator.size + edges[:,1]
    idx_sort = np.argsort(s)
    edges = edges[idx_sort,:]
    edges,idx_unique =  np.unique(edges, axis=0, return_index=True)
    A = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape=(graph_indicator.size, graph_indicator.size))
    
    x = np.loadtxt("node_attributes.txt", delimiter=",")
    edge_attr = np.loadtxt("edge_attributes.txt", delimiter=",")
    edge_attr = np.vstack((edge_attr,edge_attr))
    edge_attr = edge_attr[idx_sort,:]
    edge_attr = edge_attr[idx_unique,:]
    
    adj = []
    features = []
    edge_features = []
    idx_n = 0
    idx_m = 0
    for i in range(graph_size.size):
        adj.append(A[idx_n:idx_n+graph_size[i],idx_n:idx_n+graph_size[i]])
        edge_features.append(edge_attr[idx_m:idx_m+adj[i].nnz,:])
        features.append(x[idx_n:idx_n+graph_size[i],:])
        idx_n += graph_size[i]
        idx_m += adj[i].nnz

    return adj, features, edge_features

def normalize_adjacency(A):
    """
    Function that normalizes an adjacency matrix
    """
    n = A.shape[0]
    A += sp.identity(n)
    degs = A.dot(np.ones(n))
    inv_degs = np.power(degs, -1)
    D = sp.diags(inv_degs)
    A_normalized = D.dot(A)

    return A_normalized

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Function that converts a Scipy sparse matrix to a sparse Torch tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class GNN(nn.Module):
    """
    Simple message passing model that consists of 2 message passing layers
    and the sum aggregation function
    """
    def __init__(self, input_dim, input_seq_dim, hidden_dim, dropout, n_class):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, n_class)
        self.fc_seq1 = nn.Linear(input_seq_dim, 2*hidden_dim)
        self.fc_seq2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(2*hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_in, adj, idx, seq):

        #print(seq.shape)
        # first message passing layer
        x = self.fc1(x_in)
        x = self.relu(torch.mm(adj, x))
        x = self.dropout(x)

        # second message passing layer
        x = self.fc2(x)
        x = self.relu(torch.mm(adj, x))
        
        # sum aggregator
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out_g = torch.zeros(torch.max(idx)+1, x.size(1)).to(x_in.device)
        out_g = out_g.scatter_add_(0, idx, x)
        
        # batch normalization layer
        out_g = self.bn1(out_g)

        # mlp to produce output
        out_g = self.relu(self.fc3(out_g))
        out_g = self.dropout(out_g)

        # Processing of the sequence
        out_seq = self.relu(self.fc_seq1(seq))
        out_seq = self.dropout(out_seq)
        out_seq = self.relu(self.fc_seq2(out_seq))

        out_seq = self.bn2(out_seq)

        #print(out_g.shape)
        #print(out_seq.shape)

        # Merging both
        out = self.relu(self.fc_out(torch.cat((out_g, out_seq),1)))
        out = self.dropout(out)

        #print(torch.cat((out_g, out_seq),1).shape)

        # Produce logits
        out = self.fc4(out)

        return F.log_softmax(out, dim=1)

class EarlyStopping:
    def __init__(self, patience=1, min_delta=1, file_name="model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.file_name = file_name

    def early_stop(self, model, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            print(f"Saving best model at: '{self.file_name}'")
            torch.save(model.state_dict(), self.file_name)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
#early_stopper = EarlyStopper(patience=3, min_delta=10)
#torch.save(model.state_dict(), 'best-model-parameters.pt') # official 

# To load
#the_model = TheModelClass(*args, **kwargs)
#the_model.load_state_dict(torch.load(PATH))

# from https://mlabonne.github.io/blog/gin/
class GIN2Graphs(torch.nn.Module):
    """GIN2Graphs"""
    def __init__(self, input_dim, hidden_dim, dropout, n_class):
        super(GIN2Graphs, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(input_dim, hidden_dim),
                       BatchNorm1d(hidden_dim), ReLU(),
                       Linear(hidden_dim, hidden_dim), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(),
                       Linear(hidden_dim, hidden_dim), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(),
                       Linear(hidden_dim, hidden_dim), ReLU()))
        

        self.conv4 = GINConv(
            Sequential(Linear(input_dim, hidden_dim),
                       BatchNorm1d(hidden_dim), ReLU(),
                       Linear(hidden_dim, hidden_dim), ReLU()))
        self.conv5 = GINConv(
            Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(),
                       Linear(hidden_dim, hidden_dim), ReLU()))
        self.conv6 = GINConv(
            Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(),
                       Linear(hidden_dim, hidden_dim), ReLU()))
        
        self.lin1 = Linear(hidden_dim*6, hidden_dim*6)
        self.lin2 = Linear(hidden_dim*6, n_class)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index_distance_based,edge_index_bond_based, batch):
        # First Graph info
        h1 = self.conv1(x, edge_index_distance_based)
        h2 = self.conv2(h1, edge_index_distance_based)
        h3 = self.conv3(h2, edge_index_distance_based)

        h4 = self.conv4(x, edge_index_bond_based)
        h5 = self.conv5(h4, edge_index_bond_based)
        h6 = self.conv6(h5, edge_index_bond_based)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Graph-level readout
        h4 = global_add_pool(h4, batch)
        h5 = global_add_pool(h5, batch)
        h6 = global_add_pool(h6, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3,h4,h5,h6), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = self.dropout(h)
        h = self.lin2(h)
        
        return h, F.log_softmax(h, dim=1)

class EmbeddedEncoder(nn.Module):
    """
    Simple message passing model that consists of 2 message passing layers
    and the sum aggregation function
    """
    def __init__(self, input_dim, input_seq_dim, hidden_dim, dropout, n_class):
        super().__init__()
        #self.fc1 = nn.Linear(input_dim, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        #self.fc4 = nn.Linear(hidden_dim, n_class)
        self.fc_seq1 = nn.Linear(input_seq_dim, 2*hidden_dim)
        self.fc_seq2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_seq3 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc_seq4 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.fc_out = nn.Linear(hidden_dim//4, n_class)
        self.bn = nn.BatchNorm1d(hidden_dim//4)
        self.bn_i = nn.BatchNorm1d(input_seq_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):

        # Processing of the sequence
        seq = self.bn_i(seq)
        out_seq = self.relu(self.fc_seq1(seq))
        out_seq = self.dropout(out_seq)
        out_seq = self.relu(self.fc_seq2(out_seq))
        out_seq = self.dropout(out_seq)
        out_seq = self.relu(self.fc_seq3(out_seq))
        out_seq = self.dropout(out_seq)
        out_seq = self.relu(self.fc_seq4(out_seq))
        out_seq = self.dropout(out_seq)

        out_seq = self.bn(out_seq)

        # Merging both
        out = self.fc_out(out_seq)

        return F.log_softmax(out, dim=1)

class SequenceEncoder(nn.Module):
    """
    Simple message passing model that consists of 2 message passing layers
    and the sum aggregation function
    """
    def __init__(self, input_dim, input_seq_dim, hidden_dim, dropout, n_class):
        super(SequenceEncoder, self).__init__()
        #self.fc1 = nn.Linear(input_dim, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        #self.fc4 = nn.Linear(hidden_dim, n_class)
        self.fc_seq1 = nn.Linear(input_seq_dim, 2*hidden_dim)
        self.fc_seq2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_seq3 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc_out = nn.Linear(hidden_dim//2, n_class)
        self.bn = nn.BatchNorm1d(hidden_dim//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):

        # Processing of the sequence
        out_seq = self.relu(self.fc_seq1(seq))
        out_seq = self.dropout(out_seq)
        out_seq = self.relu(self.fc_seq2(out_seq))
        out_seq = self.dropout(out_seq)
        out_seq = self.relu(self.fc_seq3(out_seq))
        out_seq = self.dropout(out_seq)

        out_seq = self.bn(out_seq)

        # Merging both
        out = self.fc_out(out_seq)

        return F.log_softmax(out, dim=1)

class MergedModel(nn.Module):
    """
    Simple message passing model that consists of 2 message passing layers
    and the sum aggregation function
    """
    def __init__(self, input_dim_GIN, input_seq_dim_embed, hidden_dim_GIN, hidden_dim_embed, dropout, n_class):
        super().__init__()
        #self.fc1 = nn.Linear(input_dim, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        #self.fc4 = nn.Linear(hidden_dim, n_class)
        #self.fc_seq1_ngram = nn.Linear(input_seq_dim_ngram, 2*hidden_dim_ngram)
        #self.fc_seq2_ngram = nn.Linear(2*hidden_dim_ngram, hidden_dim_ngram)
        #self.fc_seq3_ngram = nn.Linear(hidden_dim_ngram, hidden_dim_ngram//2)
        #self.fc_out = nn.Linear(hidden_dim_ngram//2, n_class)
        #self.bn_ngram = nn.BatchNorm1d(hidden_dim_ngram//2)

        self.conv1 = GINConv(
            Sequential(Linear(input_dim_GIN, hidden_dim_GIN),
                       BatchNorm1d(hidden_dim_GIN), ReLU(),
                       Linear(hidden_dim_GIN, hidden_dim_GIN), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(hidden_dim_GIN, hidden_dim_GIN), BatchNorm1d(hidden_dim_GIN), ReLU(),
                       Linear(hidden_dim_GIN, hidden_dim_GIN), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(hidden_dim_GIN, hidden_dim_GIN), BatchNorm1d(hidden_dim_GIN), ReLU(),
                       Linear(hidden_dim_GIN, hidden_dim_GIN), ReLU()))
        

        self.conv4 = GINConv(
            Sequential(Linear(input_dim_GIN, hidden_dim_GIN),
                       BatchNorm1d(hidden_dim_GIN), ReLU(),
                       Linear(hidden_dim_GIN, hidden_dim_GIN), ReLU()))
        self.conv5 = GINConv(
            Sequential(Linear(hidden_dim_GIN, hidden_dim_GIN), BatchNorm1d(hidden_dim_GIN), ReLU(),
                       Linear(hidden_dim_GIN, hidden_dim_GIN), ReLU()))
        self.conv6 = GINConv(
            Sequential(Linear(hidden_dim_GIN, hidden_dim_GIN), BatchNorm1d(hidden_dim_GIN), ReLU(),
                       Linear(hidden_dim_GIN, hidden_dim_GIN), ReLU()))
        
        self.lin1 = Linear(hidden_dim_GIN*6, hidden_dim_GIN*6)
        self.lin2 = Linear(hidden_dim_GIN*6, n_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc_seq1_embed = nn.Linear(input_seq_dim_embed, 2*hidden_dim_embed)
        self.fc_seq2_embed = nn.Linear(2*hidden_dim_embed, hidden_dim_embed)
        self.fc_seq3_embed = nn.Linear(hidden_dim_embed, hidden_dim_embed//2)
        self.fc_seq4_embed = nn.Linear(hidden_dim_embed//2, hidden_dim_embed //4)
        self.fc_out_embed = nn.Linear(hidden_dim_embed//4, n_class)
        self.bn_embed = nn.BatchNorm1d(hidden_dim_embed//4)
        self.bn_i_embed = nn.BatchNorm1d(input_seq_dim_embed)

        self.fc_out1 = nn.Linear(hidden_dim_GIN*6 + hidden_dim_embed //4 + 2*n_class, 128)
        self.fc_out2 = nn.Linear(128, n_class)

    def forward(self, x, edge_index_distance_based,edge_index_bond_based, batch, embed):

        # Processing of the Graph
        # First Graph info
        h1 = self.conv1(x, edge_index_distance_based)
        h2 = self.conv2(h1, edge_index_distance_based)
        h3 = self.conv3(h2, edge_index_distance_based)

        h4 = self.conv4(x, edge_index_bond_based)
        h5 = self.conv5(h4, edge_index_bond_based)
        h6 = self.conv6(h5, edge_index_bond_based)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Graph-level readout
        h4 = global_add_pool(h4, batch)
        h5 = global_add_pool(h5, batch)
        h6 = global_add_pool(h6, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3,h4,h5,h6), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = self.dropout(h)
        h2 = self.lin2(h)


        # Processing Embeddings

        seq_embed = self.bn_i_embed(embed)
        out_embed = self.relu(self.fc_seq1_embed(seq_embed))
        #out_embed = self.dropout(out_embed)
        out_embed = self.relu(self.fc_seq2_embed(out_embed))
        #out_embed = self.dropout(out_embed)
        out_embed = self.relu(self.fc_seq3_embed(out_embed))
        #out_embed = self.dropout(out_embed)
        out_embed = self.relu(self.fc_seq4_embed(out_embed))
        out_embed = self.dropout(out_embed)

        out_embed = self.bn_embed(out_embed)

        # Merging both
        out_embed_2 = self.fc_out_embed(out_embed)

        # Merging both
        out = self.fc_out1(torch.cat((h,h2,out_embed,out_embed_2),1))
        out = self.dropout(self.relu(out))
        out = self.fc_out2(out)

        return F.log_softmax(out, dim=1)

class GNNSimple(nn.Module):
    """
    Simple message passing model that consists of 2 message passing layers
    and the sum aggregation function
    """
    def __init__(self, input_dim, input_seq_dim, hidden_dim, dropout, n_class):
        super(GNNSimple, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, n_class)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_in, adj, idx, seq):
        # first message passing layer
        x = self.fc1(x_in)
        x = self.relu(torch.mm(adj, x))
        x = self.dropout(x)

        # second message passing layer
        x = self.fc2(x)
        x = self.relu(torch.mm(adj, x))
        
        # sum aggregator
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx)+1, x.size(1)).to(x_in.device)
        out = out.scatter_add_(0, idx, x)
        
        # batch normalization layer
        out = self.bn(out)

        # mlp to produce output
        out = self.relu(self.fc3(out))
        out = self.dropout(out)
        out = self.fc4(out)

        return F.log_softmax(out, dim=1)

# Load graphs
adj, features, edge_features = load_data()

# Normalize adjacency matrices
adj = [normalize_adjacency(A) for A in adj]

# Split data into training and test sets
adj_train = list()
features_train = list()
edges_features_train= list()
y_train = list()
adj_test = list()
features_test = list()
proteins_test = list()
edges_features_test = list()

from sklearn.preprocessing import RobustScaler
transformer = RobustScaler()
transformer_amino = RobustScaler()
transformer_dist = RobustScaler()
all_coords = []
all_features = []
all_dist = []
with open('graph_labels.txt', 'r') as f:
    for i,line in enumerate(f):
        t = line.split(',')
        if len(t[1][:-1]) == 0:
            proteins_test.append(t[0])
            adj_test.append(adj[i])
            features_test.append(features[i])
            edges_features_test.append(edge_features[i])
        else:
            adj_train.append(adj[i])
            #features_train.append(features[i])
            y_train.append(int(t[1][:-1]))

            feat = features[i]
            efeat = edge_features[i]
            all_coords.append(feat[:,0:3])
            all_features.append(feat[:,25:])
            all_dist.append(efeat[:,0])
            features_train.append(feat)
            edges_features_train.append(efeat)
all_coords = np.concatenate(all_coords)
all_features = np.concatenate(all_features)
all_dist = np.concatenate(all_dist)
transformer.fit(all_coords)
transformer_amino.fit(all_features)
transformer_dist.fit(all_dist.reshape(-1,1))

def normalize_features(feat,eps=1e-7):
  res = np.copy(feat)
  ##########
  #Method 1
  # res[:,0:3] -= MIN_COORDS
  # res[:,0:3] /= (MAX_COORDS-MIN_COORDS)
  # Method 2
  # res[:,0:3] -= MEAN_COORDS
  #Method 3
  # coords = res[:,0:3]
  # tmp_min,tmp_max = np.min(coords,axis=0),np.max(coords,axis=0)
  # res[:,0:3] -= tmp_min
  # res[:,0:3] /= (tmp_max-tmp_min + eps)
  #Method 4
  # coords = res[:,0:3]
  # tmp_mean = np.mean(coords,axis=0)
  # res[:,0:3] -= tmp_mean
  #Method 5
  res[:,0:3] = transformer.transform(np.clip(res[:,0:3],-250,250))
  res[:,25:] = transformer_amino.transform(res[:,25:])
  #################
  return res

def normalize_edges(efeat):
  res = np.copy(efeat)
  res[:,0] = transformer_dist.transform(np.clip(res[:,0],0,10).reshape(-1,1)).flatten()
  return res

def separate_edges(edge_index,edge_features):
  # print(edge_index)
  idx_bond = np.where(np.logical_or(edge_features[:,2],edge_features[:,4]))[0] #peptide or hydrogen
  idx_distance = np.where(np.logical_or(edge_features[:,1],edge_features[:,3]))[0] #distance

  return edge_index[:,idx_bond],edge_index[:,idx_distance]

#from sklearn.model_selection import train_test_split
#X_train_sequence, X_val_sequence, adj_train, adj_val, features_train, features_val, y_train, y_val = train_test_split(X_train_sequence, adj_train,features_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

from sklearn.model_selection import train_test_split
#X_train_sequence, X_val_sequence, X_train_sequence_ngram, X_val_sequence_ngram, y_train, y_val = train_test_split(X_train_sequence, X_train_sequence_ngram, y_train, test_size=0.2, random_state=42, stratify=y_train)
X_train_sequence, X_val_sequence, \
adj_train, adj_val, \
features_train, features_val, \
edges_features_train, edges_features_val,\
y_train, y_val = train_test_split(X_train_sequence,adj_train,features_train,
                                  edges_features_train, y_train, 
                                  test_size=0.2, random_state=42, stratify=y_train)

for i in range(len(features_train)):
  features_train[i] = normalize_features(features_train[i])
for i in range(len(features_val)):
  features_val[i] = normalize_features(features_val[i])
for i in range(len(features_test)):
  features_test[i] = normalize_features(features_test[i])
for i in range(len(edges_features_train)):
  edges_features_train[i] = normalize_edges(edges_features_train[i])
for i in range(len(edges_features_val)):
  edges_features_val[i] = normalize_edges(edges_features_val[i])
for i in range(len(edges_features_test)):
  edges_features_test[i] = normalize_edges(edges_features_test[i])

# Initialize device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
epochs = 1000 #50
batch_size = 64 #128 #128 #64
n_hidden = 5000 #2048#1024 #128 #64 #256 #128 #64
n_input = 86
dropout = 0.3
learning_rate = 5e-5 #0.001
n_class = 18

# Compute number of training and test samples
N_train = X_train_sequence.shape[0]
N_val = X_val_sequence.shape[0]
N_test = X_test_sequence.shape[0]

# Initializes model and optimizer
#model = SequenceEncoder(n_input, input_seq_dim, n_hidden, dropout, n_class).to(device)
#model.load_state_dict(torch.load("sequence-bigger7-normed.pt"))
new_model = MergedModel(input_dim_GIN=86, input_seq_dim_embed=1024, hidden_dim_GIN=16, hidden_dim_embed=5000, dropout=dropout, n_class=n_class)
model = EmbeddedEncoder(86,1024,5000,0.3,18)
model.load_state_dict(torch.load("sequence-bigger7-normed.pt"))

new_model.fc_seq1_embed = model.fc_seq1
new_model.fc_seq2_embed = model.fc_seq2
new_model.fc_seq3_embed = model.fc_seq3
new_model.fc_seq4_embed = model.fc_seq4
new_model.bn_embed = model.bn
new_model.bn_i_embed = model.bn_i
new_model.fc_out_embed = model.fc_out


model = GIN2Graphs(input_dim=86, hidden_dim=16, dropout=0.2, n_class=18)
model.load_state_dict(torch.load("2graph_nhidden16_good.pt"))

new_model.conv1 = model.conv1
new_model.conv2 = model.conv2
new_model.conv3 = model.conv3
new_model.conv4 = model.conv4
new_model.conv5 = model.conv5
new_model.conv6 = model.conv6
new_model.lin1 = model.lin1
new_model.lin2 = model.lin2

model = new_model
del new_model
#model.load_state_dict(torch.load("merged-sequence-graph3.pt"))
model = model.to(device)

for n, p in model.named_parameters():
  if 'fc_seq1_embed' in n or 'fc_seq2_embed' in n or 'fc_seq3_embed' in n or 'fc_seq4_embed' in n or 'bn_embed' in n or 'bn_i_embed' in n or 'conv1' in n or 'conv2' in n or 'conv3' in n or 'conv4' in n or 'conv5' in n or 'conv6' in n or 'lin1' in n or 'fc_out_embed' in n or 'lin2' in n:
    p.requires_grad = False

optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
loss_function = nn.CrossEntropyLoss()
early_stopping = EarlyStopping(patience=20, min_delta=0.1, file_name="merged-sequence-graph6.pt")

# Train model
for epoch in range(epochs):
    t = time.time()
    model.train()
    train_loss = 0
    correct = 0
    count = 0
    # Iterate over the batches
    for i in range(0, N_train, batch_size):
        y_batch = list()
        X_sequence_batch = list()
        adj_batch = list()
        features_batch = list()
        edges_features_batch = list()
        idx_batch = list()
        
        # Create tensors
        for j in range(i, min(N_train, i+batch_size)):
            y_batch.append(y_train[j])
            X_sequence_batch.append(X_train_sequence[j, :])
            n = adj_train[j].shape[0]
            adj_batch.append(adj_train[j]+sp.identity(n))
            features_batch.append(features_train[j])
            edges_features_batch.append(edges_features_train[j])
            idx_batch.extend([j-i]*n)

        X_sequence_batch = np.vstack(X_sequence_batch)
        adj_batch = sp.block_diag(adj_batch)
        features_batch = np.vstack(features_batch)
        edges_features_batch = np.vstack(edges_features_batch)
        adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch).to(device).to_dense()
        # To get the edge_index representation from the adj matrix
        edge_index = adj_batch.nonzero().t().contiguous()

        y_batch = torch.LongTensor(y_batch).to(device)
        X_sequence_batch = torch.FloatTensor(X_sequence_batch).to(device)
        edge_index_bond, edge_index_distance = separate_edges(edge_index,edges_features_batch)
        features_batch = torch.FloatTensor(features_batch).to(device)
        idx_batch = torch.LongTensor(idx_batch).to(device)
        
        optimizer.zero_grad()
        output = model(features_batch, edge_index_distance,edge_index_bond, idx_batch, X_sequence_batch)
        loss = loss_function(output, y_batch)
        train_loss += loss.item() * output.size(0)
        count += output.size(0)
        preds = output.max(1)[1].type_as(y_batch)
        correct += torch.sum(preds.eq(y_batch).double())
        loss.backward()
        optimizer.step()
    
    #if epoch % 5 == 0:
    #print('Epoch: {:03d}'.format(epoch+1),
    #      'loss_train: {:.4f}'.format(train_loss / count),
    #      'acc_train: {:.4f}'.format(correct / count),
    #      'time: {:.4f}s'.format(time.time() - t))
    loss_train = train_loss / count
    acc_train = correct / count
    time_train = time.time() - t

    # Validate

    model.eval()
    val_loss = 0
    correct = 0
    count = 0
    # Iterate over the batches
    for i in range(0, N_val, batch_size):
        y_batch = list()
        X_sequence_batch = list()
        adj_batch = list()
        features_batch = list()
        edges_features_batch = list()
        idx_batch = list()
        
        # Create tensors
        for j in range(i, min(N_val, i+batch_size)):
            y_batch.append(y_val[j])
            X_sequence_batch.append(X_val_sequence[j, :])
            n = adj_val[j].shape[0]
            adj_batch.append(adj_val[j]+sp.identity(n))
            features_batch.append(features_val[j])
            edges_features_batch.append(edges_features_val[j])
            idx_batch.extend([j-i]*n)

        X_sequence_batch = np.vstack(X_sequence_batch)
        adj_batch = sp.block_diag(adj_batch)
        features_batch = np.vstack(features_batch)
        edges_features_batch = np.vstack(edges_features_batch)
        adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch).to(device).to_dense()
        # To get the edge_index representation from the adj matrix
        edge_index = adj_batch.nonzero().t().contiguous()

        y_batch = torch.LongTensor(y_batch).to(device)
        X_sequence_batch = torch.FloatTensor(X_sequence_batch).to(device)
        edge_index_bond, edge_index_distance = separate_edges(edge_index,edges_features_batch)
        features_batch = torch.FloatTensor(features_batch).to(device)
        idx_batch = torch.LongTensor(idx_batch).to(device)
        
        output = model(features_batch, edge_index_distance,edge_index_bond, idx_batch, X_sequence_batch)
        loss = loss_function(output, y_batch)
        val_loss += loss.item() * output.size(0)
        count += output.size(0)
        preds = output.max(1)[1].type_as(y_batch)
        correct += torch.sum(preds.eq(y_batch).double())

    loss_val = val_loss / count
    print('Epoch: {:03d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train),
          'loss_val: {:.4f}'.format(loss_val),
          'acc_train: {:.4f}'.format(acc_train),
          'acc_val: {:.4f}'.format(correct / count),
          'time_train: {:.4f}s'.format(time_train))
    if early_stopping.early_stop(model, loss_val):
      break

!cp "./merged-sequence-graph4.pt" "/content/drive/MyDrive/Master MVA/ALTEGRAD/Data Challenge ALTEGRAD/Files/"

!ls -lh *

model.load_state_dict(torch.load("merged-sequence-graph4.pt"))

# Evaluate model
model.eval()
y_pred_proba = list()
# Iterate over the batches
for i in range(0, N_test, batch_size):
    
    y_batch = list()
    X_test_sequence_batch = list()
    adj_batch = list()
    features_batch = list()
    edges_features_batch = list()
    idx_batch = list()
    
    # Create tensors
    for j in range(i, min(N_test, i+batch_size)):
        X_test_sequence_batch.append(X_test_sequence[j, :])
        n = adj_test[j].shape[0]
        adj_batch.append(adj_test[j]+sp.identity(n))
        features_batch.append(features_test[j])
        edges_features_batch.append(edges_features_test[j])
        idx_batch.extend([j-i]*n)

    X_test_sequence_batch = np.vstack(X_test_sequence_batch)
    adj_batch = sp.block_diag(adj_batch)
    features_batch = np.vstack(features_batch)
    edges_features_batch = np.vstack(edges_features_batch)
    adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch).to(device).to_dense()
    # To get the edge_index representation from the adj matrix
    edge_index = adj_batch.nonzero().t().contiguous()

    #X_test_sequence_batch = sparse_mx_to_torch_sparse_tensor(X_test_sequence_batch).to(device)
    X_test_sequence_batch = torch.FloatTensor(X_test_sequence_batch).to(device)
    edge_index_bond, edge_index_distance = separate_edges(edge_index,edges_features_batch)
    features_batch = torch.FloatTensor(features_batch).to(device)
    idx_batch = torch.LongTensor(idx_batch).to(device)

    output = model(features_batch, edge_index_distance,edge_index_bond, idx_batch, X_test_sequence_batch)
    y_pred_proba.append(output)
    
y_pred_proba = torch.cat(y_pred_proba, dim=0)
y_pred_proba = torch.exp(y_pred_proba)
y_pred_proba = y_pred_proba.detach().cpu().numpy()

# Write predictions to a file
with open('sample_submission.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = list()
    for i in range(18):
        lst.append('class'+str(i))
    lst.insert(0, "name")
    writer.writerow(lst)
    for i, protein in enumerate(proteins_test):
        lst = y_pred_proba[i,:].tolist()
        lst.insert(0, protein)
        writer.writerow(lst)

model.fc1

from sklearn.feature_extraction.text import TfidfVectorizer

from yellowbrick.text import TSNEVisualizer
from yellowbrick.datasets import load_hobbies

# Load the data and create document vectors

# Create the visualizer and draw the vectors
tsne = TSNEVisualizer()
tsne.fit(X_train_sequence, y_train)
tsne.show()

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import random
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt