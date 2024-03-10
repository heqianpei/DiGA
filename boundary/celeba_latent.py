import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import Dataset, Subset
import pdb


class CelebA_latent(data.Dataset):
    def __init__(self, img_path="/home/heqianpei/CelebA/img_align_celeba/", attr_path="/home/heqianpei/CelebA/list_attr_celeba.csv",
                 metadata_path='/home/heqianpei/CelebA/metadata.csv', image_size=224, mode="train", target="Smiling", sensitive="Male"):
        super(CelebA_latent, self).__init__()
        self.img_path = img_path
        self.df = pd.read_csv(attr_path)
        self.df_meta = pd.read_csv(metadata_path)
        self.images = []
        self.labels_target = []
        self.labels_sensitive = []
        if mode == 'train':
            # txt_file=open(img_path,'r')
            # img_list = txt_file.readlines()
            for i in range(len(self.df_meta)):
                # if (self.df_meta.loc[i]['partition'] == 0) & (self.df_meta.loc[i]['image_id'][:6] in img_list):
                if (self.df_meta.loc[i]['partition'] == 0) :
                    self.images.append(self.df_meta.loc[i]['image_id'])
                    y = int((self.df.loc[i][target]+1)/2)
                    self.labels_target.append(y)
                    s = int((self.df.loc[i][sensitive]+1)/2)
                    self.labels_sensitive.append(s)
        if mode == 'valid':
            for i in range(len(self.df_meta)):
                if self.df_meta.loc[i]['partition'] == 1:
                    self.images.append(self.df_meta.loc[i]['image_id'])
                    y = int((self.df.loc[i][target]+1)/2)
                    self.labels_target.append(y)
                    s = int((self.df.loc[i][sensitive]+1)/2)
                    self.labels_sensitive.append(s)
        if mode == 'test':
            for i in range(len(self.df_meta)):
                if self.df_meta.loc[i]['partition'] == 2:
                    self.images.append(self.df_meta.loc[i]['image_id'])
                    y = int((self.df.loc[i][target]+1)/2)
                    self.labels_target.append(y)
                    s = int((self.df.loc[i][sensitive]+1)/2)
                    self.labels_sensitive.append(s)
                   
        self.length = len(self.images)
        
    def __getitem__(self, index):
        last_dot_index = self.images[index].rfind('.')
        filename = self.images[index][:last_dot_index]
        img = torch.load(os.path.join(self.img_path, filename+".pth"))
        y = torch.tensor(self.labels_target[index])
        s = torch.tensor(self.labels_sensitive[index])
        img_name = self.images[index]
        return img, y, s, img_name
    def __len__(self):
        return self.length
    
class CelebA_latent_multi(data.Dataset):
    def __init__(self, img_path="/home/zhangfengda/CelebA/align_latent_code_Smiling_Young_Male_20000_0.111_traindata/", attr_path="/home/zhangfengda/CelebA/list_attr_celeba_Smiling_Young_Male_20000_0.111.csv",
                 metadata_path='/home/zhangfengda/CelebA/metadata_Smiling_Young_Male_20000_0.111.csv', image_size=224, mode="train", target1="Smiling",target2="Young",sensitive="Male",choose=0):
        super(CelebA_latent_multi, self).__init__()
        self.img_path = img_path
        self.df = pd.read_csv(attr_path)
        self.df_meta = pd.read_csv(metadata_path)
        self.images = []
        self.labels_target = []
        self.labels_sensitive = []
        self.choose = choose
        if mode == 'train':
            # txt_file=open(img_path,'r')
            # img_list = txt_file.readlines()
            for i in range(len(self.df_meta)):
                # if (self.df_meta.loc[i]['partition'] == 0) & (self.df_meta.loc[i]['image_id'][:6] in img_list):
                if (self.df_meta.loc[i]['partition'] == 0) :
                    self.images.append(self.df_meta.loc[i]['image_id'])
                    y = int((self.df.loc[i][target1]+1)/2)*2+int((self.df.loc[i][target2]+1)/2)
                    self.labels_target.append(y)
                    s = int((self.df.loc[i][sensitive]+1)/2)
                    self.labels_sensitive.append(s)
        if mode == 'valid':
            for i in range(len(self.df_meta)):
                if self.df_meta.loc[i]['partition'] == 1:
                    self.images.append(self.df_meta.loc[i]['image_id'])
                    y = int((self.df.loc[i][target1]+1)/2)*2+int((self.df.loc[i][target2]+1)/2)
                    self.labels_target.append(y)
                    s = int((self.df.loc[i][sensitive]+1)/2)
                    self.labels_sensitive.append(s)
        if mode == 'test':
            for i in range(len(self.df_meta)):
                if self.df_meta.loc[i]['partition'] == 2:
                    self.images.append(self.df_meta.loc[i]['image_id'])
                    y = int((self.df.loc[i][target1]+1)/2)*2+int((self.df.loc[i][target2]+1)/2)
                    self.labels_target.append(y)
                    s = int((self.df.loc[i][sensitive]+1)/2)
                    self.labels_sensitive.append(s)
                   
        self.length = len(self.images)
        
    def __getitem__(self, index):
        img = torch.load(os.path.join(self.img_path, self.images[index][:6]+".pth"))
        if self.labels_target[index]==self.choose:
            y = 1
        else:
            y = 0
        # y = torch.tensor(self.labels_target[index])
        s = torch.tensor(self.labels_sensitive[index])
        img_name = self.images[index]
        return img, y, s, img_name
    def __len__(self):
        return self.length
    

