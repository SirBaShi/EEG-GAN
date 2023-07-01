import torch
import torchvision
import os
import numpy as np
import scipy.io
from torch.utils.data import Dataset
import torch.nn.functional as F

class MyDataset(Dataset):
    def __init__(self,data = 32):
        self.file_path32 = './32-DOC-all/'
        self.file_path64 = './64-DOC-all/'
        self.label_path32 = './label32/'
        self.label_path64 = './label64/'
        if data == 32:
            self.file_path = self.file_path32
            self.label_path = self.label_path32
        else:
            self.file_path = self.file_path64
            self.label_path = self.label_path64
        self.data_files = [f for f in os.listdir(self.file_path)]
        self.data = [torch.tensor(scipy.io.loadmat(os.path.join(self.file_path,f))['data']) for f in self.data_files]
        self.label_files = [f for f in os.listdir(self.label_path)]
        self.label = []
        for f in self.label_files:
            with open (os.path.join(self.label_path,f)) as m:
                data1 = m.read()
                self.label.append(int(data1))
    def __getitem__(self, index):
        x = self.data
        y = torch.tensor(self.label).reshape(-1,1)
        return x,y
    def __len__(self):
        return len(self.data)