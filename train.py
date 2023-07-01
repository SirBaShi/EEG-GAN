import torch
import torchvision
import os
import numpy as np
import random
from Dataset import MyDataset
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

dataset = MyDataset(data = 32)
dataloader = DataLoader(dataset = dataset, batch_size= 32, shuffle=True)

for data,label in dataloader:
    clf = XGBClassifier(booster = 'gbtree', n_estimators = 100, learning_rate = 0.001,loss = 'deviance')
    clf.fit(data,label)
    index1 = random.sample(range(0,32),sample_num=6)
    test_data = [data[index1]]
    test_label = [label[index1]]
    predict = clf.predict(test_data)
    acc = accuracy_score(test_label,predict)
    print(acc)