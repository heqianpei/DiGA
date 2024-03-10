import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import Dataset, Subset
import pdb


class CelebA(data.Dataset):
    def __init__(self, img_path="/home/heqianpei/CelebA/img_align_celeba/", attr_path="/home/heqianpei/CelebA/list_attr_celeba.csv",
                 metadata_path='/home/heqianpei/CelebA/metadata.csv', image_size=224, mode="train", target="Smiling", sensitive="Male"):
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
                    self.images.append(self.df_meta.loc[i]['image_id'][:6]+".png")
                    y = int((self.df.loc[i][target]+1)/2)
                    self.labels_target.append(y)
                    s = int((self.df.loc[i][sensitive]+1)/2)
                    self.labels_sensitive.append(s)
        if mode == 'valid':
            for i in range(len(self.df_meta)):
                if self.df_meta.loc[i]['partition'] == 1:
                    self.images.append(self.df_meta.loc[i]['image_id'][:6]+".png")
                    y = int((self.df.loc[i][target]+1)/2)
                    self.labels_target.append(y)
                    s = int((self.df.loc[i][sensitive]+1)/2)
                    self.labels_sensitive.append(s)
        if mode == 'test':
            for i in range(len(self.df_meta)):
                if self.df_meta.loc[i]['partition'] == 2:
                    self.images.append(self.df_meta.loc[i]['image_id'][:6]+".png")
                    y = int((self.df.loc[i][target]+1)/2)
                    self.labels_target.append(y)
                    s = int((self.df.loc[i][sensitive]+1)/2)
                    self.labels_sensitive.append(s)
        
        # self.tf = transforms.Compose([
        #     transforms.CenterCrop(178),
        #     transforms.Resize(image_size),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # ])  
        # self.tf = transforms.Compose([
		# 		transforms.Resize((256, 256)),
		# 		transforms.ToTensor(),
		# 		transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])          
        self.length = len(self.images)
        
    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.img_path, self.images[index])))
        # img = Image.open(os.path.join(self.img_path, self.images[index]))
        y = torch.tensor(self.labels_target[index])
        s = torch.tensor(self.labels_sensitive[index])
        img_name = self.images[index]
        return img, y, s, img_name
    def __len__(self):
        return self.length

# dataset = CelebA(img_path="/home/heqianpei/CelebA/img_align_celeba_y=Smiling_s=Male_size=20000_bias=0.25/", attr_path="/home/heqianpei/CelebA/list_attr_celeba_y=Smiling_s=Male_size=20000_bias=0.25.csv",
#                  metadata_path='/home/heqianpei/CelebA/metadata_y=Smiling_s=Male_size=20000_bias=0.25.csv')
# pdb.set_trace()
