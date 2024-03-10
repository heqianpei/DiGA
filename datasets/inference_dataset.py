from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import pandas as pd
import torch
import os
import pdb


class InferenceDataset(Dataset):

	def __init__(self, root, opts, attr_path="/home/heqianpei/CelebA/list_attr_celeba.csv",
                 metadata_path='/home/heqianpei/CelebA/metadata.csv', target="Smiling", sensitive="Male",transform=None, preprocess=None,mode = 1):
		# self.paths = sorted(data_utils.make_dataset(root))
		self.transform = transform
		self.preprocess = preprocess
		self.opts = opts

		self.paths = root
		self.mode = mode
		if self.mode == 1:
			self.df = pd.read_csv(attr_path)
			self.df_meta = pd.read_csv(metadata_path)
			self.images = []
			self.labels_target = []
			self.labels_sensitive = []
			for i in range(len(self.df_meta)):
				if (self.df_meta.loc[i]['partition'] == 0) & (os.path.isfile(os.path.join(self.paths, self.df_meta.loc[i]['image_id']))):
					self.images.append(self.df_meta.loc[i]['image_id'])
					y = int((self.df.loc[i][target]+1)/2)
					self.labels_target.append(y)
					s = int((self.df.loc[i][sensitive]+1)/2)
					self.labels_sensitive.append(s)
		

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		from_path = os.path.join(self.paths, self.images[index])
		if self.preprocess is not None:
			from_im = self.preprocess(from_path)
		else:
			from_im = Image.open(from_path).convert('RGB')
		if self.transform:
			from_im = self.transform(from_im)
			img_name = self.images[index]
		if self.mode == 1:
			y = torch.tensor(self.labels_target[index])
			s = torch.tensor(self.labels_sensitive[index])
			return from_im, y, s, img_name
		else:
			return from_im
# dataset = InferenceDataset(img_path="/home/heqianpei/CelebA/img_align_celeba_y=Smiling_s=Male_size=20000_bias=0.25/", attr_path="/home/heqianpei/CelebA/list_attr_celeba_y=Smiling_s=Male_size=20000_bias=0.25.csv",
#                  metadata_path='/home/heqianpei/CelebA/metadata_y=Smiling_s=Male_size=20000_bias=0.25.csv')
# pdb.set_trace()

class InferenceDataset_Multi(Dataset):

	def __init__(self, root, opts, attr_path="/home/heqianpei/CelebA/list_attr_celeba.csv",
                 metadata_path='/home/heqianpei/CelebA/metadata.csv', target1="Smiling",target2="Young",sensitive="Male",transform=None, preprocess=None,mode = 1):
		# self.paths = sorted(data_utils.make_dataset(root))
		self.transform = transform
		self.preprocess = preprocess
		self.opts = opts

		self.paths = root
		self.mode = mode
		if self.mode == 1:
			self.df = pd.read_csv(attr_path)
			self.df_meta = pd.read_csv(metadata_path)
			self.images = []
			self.labels_target = []
			self.labels_sensitive = []
			for i in range(len(self.df_meta)):
				if (self.df_meta.loc[i]['partition'] == 0) & (os.path.isfile(os.path.join(self.paths, self.df_meta.loc[i]['image_id']))):
					self.images.append(self.df_meta.loc[i]['image_id'])
					y = int((self.df.loc[i][target1]+1)/2)*2+int((self.df.loc[i][target2]+1)/2)
					self.labels_target.append(y)
					s = int((self.df.loc[i][sensitive]+1)/2)
					self.labels_sensitive.append(s)


		

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		from_path = os.path.join(self.paths, self.images[index])
		if self.preprocess is not None:
			from_im = self.preprocess(from_path)
		else:
			from_im = Image.open(from_path).convert('RGB')
		if self.transform:
			from_im = self.transform(from_im)
			img_name = self.images[index]
		if self.mode == 1:
			y = torch.tensor(self.labels_target[index])
			s = torch.tensor(self.labels_sensitive[index])
			return from_im, y, s, img_name
		else:
			return from_im