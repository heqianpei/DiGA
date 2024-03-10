# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Custom datasets for CelebA and CelebA-HQ."""

# import cv2
import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import random
import pandas as pd
import pdb


class CelebA(data.Dataset):
    def __init__(self, img_path="/home/zfd/datasets/CelebA/img_align_celeba",
                 attr_path="/home/zfd/CelebA/list_attr_celeba_y=Smiling_s=Male_size=20000_bias=0.1111111111111111.csv",
                 metadata_path='/home/zfd/CelebA/metadata_y=Smiling_s=Male_size=20000_bias=0.1111111111111111.csv',
                 image_size=224, mode='train', target="Smiling", sensitive="Male"):
        super(CelebA, self).__init__()
        self.img_path = img_path
        self.df = pd.read_csv(attr_path)
        self.df_meta = pd.read_csv(metadata_path)
        self.images = []
        self.labels_target = []
        self.labels_sensitive = []
        if mode == 'train':
            for i in range(len(self.df_meta)):
                if self.df_meta.loc[i]['partition'] == 0:
                    self.images.append(self.df_meta.loc[i]['image_id'][:6]+".jpg")
                    y = int((self.df.loc[i][target]+1)/2)
                    self.labels_target.append(y)
                    s = int((self.df.loc[i][sensitive]+1)/2)
                    self.labels_sensitive.append(s)
        if mode == 'valid':
            for i in range(len(self.df_meta)):
                if self.df_meta.loc[i]['partition'] == 1:
                    self.images.append(self.df_meta.loc[i]['image_id'][:6]+".jpg")
                    y = int((self.df.loc[i][target]+1)/2)
                    self.labels_target.append(y)
                    s = int((self.df.loc[i][sensitive]+1)/2)
                    self.labels_sensitive.append(s)
        if mode == 'test':
            for i in range(len(self.df_meta)):
                if self.df_meta.loc[i]['partition'] == 2:
                    self.images.append(self.df_meta.loc[i]['image_id'][:6]+".jpg")
                    y = int((self.df.loc[i][target]+1)/2)
                    self.labels_target.append(y)
                    s = int((self.df.loc[i][sensitive]+1)/2)
                    self.labels_sensitive.append(s)

        orig_w = 178
        orig_h = 218
        orig_min_dim = min(orig_w, orig_h)

        self.tf_cl = transforms.Compose([
                transforms.CenterCrop(orig_min_dim),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
        self.tf_tr = transforms.Compose([
                transforms.CenterCrop(orig_min_dim),
                transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.tf_te = transforms.Compose([
                transforms.CenterCrop(orig_min_dim),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.length = len(self.images)
        
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.images[index]))
        x_cl = self.tf_cl(img)
        x_tr = self.tf_tr(img)
        x_te = self.tf_te(img)
        y = torch.tensor(self.labels_target[index])
        s = torch.tensor(self.labels_sensitive[index])
        img_name = self.images[index]
        return x_cl, x_tr, x_te, y, s, img_name
    
    def __len__(self):
        return self.length


class CelebA_Pair(data.Dataset):
    def __init__(self, img_path="/home/zfd/CelebA_degree/two_degree_Blond_Hair_Male_20000_0.111_0.0001_10000_different_degree_norm",
                 attr_path="/home/zfd/CelebA_degree/csv_list/list_attr_celeba_Blond_Hair_Male_20000_0.111.csv",
                 metadata_path='/home/zfd/CelebA_degree/csv_list/metadata_Blond_Hair_Male_20000_0.111.csv',
                 image_size=224, target="Blond_Hair", sensitive="Male"):
        super(CelebA_Pair, self).__init__()
        self.img_path = img_path
        self.df = pd.read_csv(attr_path)
        self.df_meta = pd.read_csv(metadata_path)
        self.images = []
        self.labels_target = []
        self.labels_sensitive = []
        for i in range(len(self.df_meta)):
            if self.df_meta.loc[i]['partition'] == 0:
                self.images.append(self.df_meta.loc[i]['image_id'])
                y = int((self.df.loc[i][target]+1)/2)
                self.labels_target.append(y)
                s = int((self.df.loc[i][sensitive]+1)/2)
                self.labels_sensitive.append(s)

        orig_w = 178
        orig_h = 218
        orig_min_dim = min(orig_w, orig_h)     
        target_resolution = (image_size, image_size)

        self.tf = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
        self.length = len(self.images)
        
    def __getitem__(self, index): 
        list_choose_pos = [3,4,5]
        list_choose_neg = [-3,-4,-5]  

        random_number_pos = list_choose_pos[random.randint(0,len(list_choose_pos)-1)]
        random_number_neg = list_choose_neg[random.randint(0,len(list_choose_neg)-1)]
        last_dot_index = self.images[index].rfind('.')
        filename = self.images[index][:last_dot_index]
        view_pos = self.tf(Image.open(os.path.join(self.img_path, filename+"_degree_"+str(random_number_pos)+".0.jpg")))
        view_neg = self.tf(Image.open(os.path.join(self.img_path, filename+"_degree_"+str(random_number_neg)+".0.jpg")))
        
        return view_pos, view_neg
    def __len__(self):
        return self.length


class CelebA_degree(data.Dataset):
    def __init__(self, img_path="/home/zfd/CelebA_degree/two_degree_Blond_Hair_Male_20000_0.111_0.0001_10000_different_degree_norm",
                 attr_path="/home/zfd/CelebA_degree/csv_list/list_attr_celeba_Blond_Hair_Male_20000_0.111.csv",
                 metadata_path='/home/zfd/CelebA_degree/csv_list/metadata_Blond_Hair_Male_20000_0.111.csv',
                 image_size=224, mode="train",target="Blond_Hair", sensitive="Male"):
        super(CelebA_degree, self).__init__()
        self.img_path = img_path
        self.df = pd.read_csv(attr_path)
        self.df_meta = pd.read_csv(metadata_path)
        self.images = []
        self.labels_target = []
        self.labels_sensitive = []
        self.mode = mode
        if mode == 'train':
            for i in range(len(self.df_meta)):
                if self.df_meta.loc[i]['partition'] == 0:
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

        orig_w = 178
        orig_h = 218
        orig_min_dim = min(orig_w, orig_h)     
        target_resolution = (image_size, image_size)
        if mode == 'train':
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.9, 1.0),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.CenterCrop(orig_min_dim),
                transforms.Resize(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
    
        self.length = len(self.images)
        
    def __getitem__(self, index):
        import random
        # img = Image.open(os.path.join(self.img_path, self.images[index]))
        y = torch.tensor(self.labels_target[index])
        s = torch.tensor(self.labels_sensitive[index])
        if self.mode == 'train':
            list_choose_0 = [4,5,6]
            list_choose_1 = [-4,-5,-6]
            
            if y == 0:
                random_num = random.randint(0,len(list_choose_0)-1)
                random_number = list_choose_0[random_num]
            elif y == 1:
                random_num = random.randint(0,len(list_choose_1)-1)
                random_number = list_choose_1[random_num]
            last_dot_index = self.images[index].rfind('.')
            filename = self.images[index][:last_dot_index]
            img = self.tf(Image.open(os.path.join(self.img_path, filename+"_degree_"+str(random_number)+".0.jpg")))
            img_name = filename+"_degree_"+str(random_number)+".0.jpg"
        else:
            img = self.tf(Image.open(os.path.join(self.img_path, self.images[index])))
            img_name = self.images[index]
        
        return img, y, s, img_name
    def __len__(self):
        return self.length
    
