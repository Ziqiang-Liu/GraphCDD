from torch import optim
from param import parameter_parser
from model import GCN
import load_data 
import torch
import pandas as pd
import random
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import math



from sklearn.metrics import roc_auc_score
#from torch import tensor

def train(model, train_data, optimizer, opt):
    model.train()

    for epoch in range(0, opt.epoch):
        model.zero_grad()
        circ_dis,circ_drug,drug_dis,cir_fea,drug_fea,dis_fea = model(train_data)
        loss1 = torch.nn.BCEWithLogitsLoss(reduction='mean')
        #print(train_data['drug_dis'].shape)
        loss1 = loss1(circ_dis, train_data['c_dis'].cuda())
        loss1.backward()
        optimizer.step()
        print("cricRNA-disease loss \t",str(epoch),"\t",loss1.item())
    
    optimizer2 = torch.optim.Adam(model.parameters(), lr=0.005)
    for epoch in range(0, opt.epoch):
        model.zero_grad()
        circ_dis,circ_drug,drug_dis,cir_fea,drug_fea,dis_fea = model(train_data)
        loss2 = torch.nn.BCEWithLogitsLoss(reduction='mean')
        #print(train_data['drug_dis'].shape)
        loss2 = loss2(drug_dis, train_data['drug_dis'].cuda())
        loss2.backward()
        optimizer2.step()
        print("drug-disease loss \t",str(epoch),"\t",loss2.item())
    optimizer3 = torch.optim.Adam(model.parameters(), lr=0.005)
    for epoch in range(0, opt.epoch):
        model.zero_grad()
        circ_dis,circ_drug,drug_dis,cir_fea,drug_fea,dis_fea = model(train_data)
        loss3 = torch.nn.BCEWithLogitsLoss(reduction='mean')
   

        loss3 = loss3(circ_drug, train_data['c_d'].cuda())

        loss3.backward()
        optimizer3.step()
        print("circRNA-drug loss \t",str(epoch),"\t",loss3.item())

    
    return model




