import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import math
import numpy as np


class GCF(nn.modules.module.Module):

    def __init__(self,K:int,in_dim,out_dim,graph_number,device,bias = False):
        super(GCF, self).__init__()
        self.K = K
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.graph_number = graph_number
        self.device = device

       
        self.w = nn.Parameter(torch.zeros(self.in_dim*K, self.out_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        self.reset_parameters()
  



    def forward(self, A,X):
   
            hidden = [X]
            for k in range(self.K-1):
                hidden.append(torch.einsum("mn,rng->rmg",[A , hidden[-1]] )) 
            out = torch.einsum("rng,gf->rnf",[hidden,self.w] )
            if self.bias is not None:
                out += self.bias
            return out



class GCNN(nn.Module):

    def __init__(self, K:int, hidden_dim:int,in_dim:int,out_dim:int):
        super(GCNN, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim  = hidden_dim
        self.out_dim = out_dim
        self.layer_number = layer_number
       
        self.device = device

        self.layer1 = []
        self.hidden_layers = []
 
        self.layer1 = GCF(K,in_dim,hidden_dim,device,bias)


        for _ in range(layer_number - 2):
            self.hidden_layers.append(GCF(K,hidden_dim,in_dim,device,bias))
             

        self.hidden_layers.append(GCF(1,hidden_dim,out_dim,device,False))
        
       
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
  
        self.K = K
        self.dropout = dropout


        self.features = features
        self.A = A
        self.L1loss = nn.SmoothL1Loss()


      
    def gnn_layer(self,X,A):
        print(X.shape,A.shape)
        hidden_1 = self.layer1(A, X)
        hidden_1 = F.relu(hidden_1)
        hidden = hidden_1
        for k in range(1, self.layer_number-1):
            hidden = self.hidden_layers[k-1](A,hidden)
           
            hidden = F.relu(hidden)
        
        out= hidden
        
        return out
    def forward(self,X = None, A = None ):

        if X is None:
            X = self.features
        if A is None:
            A = self.A
      
        hidden = self.gnn_layer(X,A)
        hidden = self.hidden_layers[-1](A,hidden)
        out= hidden
        return out

    def to_prob(self, nodes):
        scores = self.forward()
        scores = scores[nodes]
        pos_scores = torch.sigmoid(scores)
        return pos_scores
    def gnn_pertubaton(self, nodes,A_hat):
        hidden_1 = self.forward(self.features[nodes],self.A)
        hidden_2 = self.forward(self.features[nodes],A_hat)
        c1 = self.gnn_layer(self.features[nodes],self.A)
        c2 = self.gnn_layer(self.features[nodes],A_hat)


  

    
class GCNNIL2(GCNN):

    def __init__(self, K:int, hidden_dim:int,in_dim:int,out_dim:int):
        super(GCNNIL2, self).__init__(K, hidden_dim,in_dim,out_dim, layer_number,A,features,dropout ,bias,device )
    
    def ILLoss(self):
        lbds = [torch.zeros(len(self.eigvals))]
        response = [torch.zeros(len(self.eigvals))+1]
        h = torch.zeros(len(self.eigvals))+1
        for j in range(self.K-1):
            h = h*self.eigvals
            lbds.append(h*(j+1))
            response.append(h)

        lbds = torch.cat(lbds,dim=0).reshape(self.K,-1)
        response = torch.cat(response,dim = 0).reshape(self.K,-1)

        out = 0
        C = 0
        self.g = 0
        for l in range(self.layer_number-1):
          
            H = self.layer1.w.view(self.K,self.in_dim,self.hidden_dim)
           
  
            Cs = torch.einsum("km,kij->mij",[lbds, H])
            Cs = torch.abs(Cs)
            gs = torch.einsum("km,kij->mij",[response, H])
            gs = torch.abs(gs)
            C = max(C, torch.max(Cs))
            g = gs
            self.g = max(self.g,torch.max(g).item())

            out += self.L1loss(Cs,torch.zeros(Cs.shape))
            out += self.L1loss(gs,torch.zeros(gs.shape))

        self.C = C
        self.C2 = f.item()
        return out
    
   
