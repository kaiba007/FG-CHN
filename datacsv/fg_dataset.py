# Time: 2022-12-20-11-34
# Author: Xianxian Zeng
# Name: fg_dataset.py
# Details: Dataset with Pytorch-Lightning for Fine-graiend Hashing

import os
from torch.utils.data import Dataset
import PIL.Image as Image
import pandas as pd
import torch
import torch.nn.functional as F

def default_loader(path):
    return Image.open(path).convert('RGB')

class FG_dataset(Dataset):
    def __init__(self, csv_filename, root_dir=None, config=None, data_type='train', transform=None, loader=default_loader):
        self.root_dir = root_dir
        self.data_type = data_type
        self.filename = csv_filename
        self.transform = transform
        self.loader = loader
        self.config = config
        imgs = []
        labels = []
        if isinstance(self.filename, (list,tuple)): # (query, gallery) or (train,)
            for i in range(len(self.filename)):
                data_list = pd.read_csv(self.filename[i])
                for index, row in data_list.iterrows():
                    imgs.append((row['img_path'], row['label'], i))
                    labels.append(row['label'])
        self.imgs = imgs

        if self.config is not None:
            self.cal_pseudo_hashing_code(labels)


    def __getitem__(self, index):
        filename, label, flag = self.imgs[index]
        img_path = os.path.join(self.root_dir, filename)
        img = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.config is not None:
            return img, label, self.pseudo_code[index]
        else:
            return img, label, flag

    def __len__(self):
        return len(self.imgs)   

    # generate pseudo-hashing-code like FISH(TIP2022)
    def cal_pseudo_hashing_code(self, labels):
        bits = self.config.code_length
        with torch.no_grad():
            train_labels = torch.tensor(labels) if isinstance(labels, list) else labels
            # training one-hot code
            tohc = F.one_hot(train_labels).to(torch.float)
            train_size = train_labels.size(0)
            sigma = torch.tensor(1, dtype=torch.float)
            delta = torch.tensor(0.0001, dtype=torch.float)
            setting_iter = 15

            V = torch.randn(bits, train_size)
            B = torch.sign(torch.randn(bits, train_size))
            S1, E, S2 = torch.svd(torch.mm(B, V.t()))
            R = torch.mm(S1, S2)
            T = tohc.t()

            for i in range(setting_iter):
                print("Code generating with %d..." % i)
                B = -1 * torch.ones((bits, train_size))
                B[(torch.mm(R,V)) >= 0] = 1

                Ul = torch.mm(sigma * torch.mm(T, V.t()), 
                              torch.linalg.pinv(sigma * torch.mm(V, V.t())))

                V = torch.mm(torch.linalg.pinv(sigma * torch.mm(Ul.t(), Ul) + delta * torch.mm(R.t(), R)),
                              sigma * torch.mm(Ul.t(), T) + delta * torch.mm(R.t(), B))
                
                S1, E, S2 = torch.svd(torch.mm(B, V.t()))

                R = torch.mm(S1, S2)

            self.pseudo_code = torch.sign(B.t())
            print("Code generated!", "Size:", B.size())
 


