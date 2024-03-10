import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import argparse
import time
from tqdm import tqdm
from data import *
from model import *
import pdb
import random


def get_result(data_num,predict_num,right_num):
   acc = (sum(right_num)/sum(data_num))
   group_acc = [right_num[0]/data_num[0],right_num[1]/data_num[1],right_num[2]/data_num[2],right_num[3]/data_num[3]]
   group_mean_acc = sum(group_acc)/4
   worst_acc = min(group_acc)
   var = np.var(group_acc)
   DP = abs((predict_num[2]/(data_num[0]+data_num[2]))-(predict_num[3]/(data_num[1]+data_num[3])))
   EO= max(abs(group_acc[3]-group_acc[2]),abs(group_acc[1]-group_acc[0]))

   return acc,group_mean_acc,worst_acc,var,DP,EO

def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)
seed_everything(0)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=(5, 5), stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=(5, 5), stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64*3*3, 2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


def train_model(encoder, classifier, criterion, optimizer, num_epochs=50):
  best_acc = 0
  for epoch in range(num_epochs):
    encoder.eval()
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 10)

    for mode in ['train', 'val']:
      if mode == 'train':
        classifier.train()
      else:
        classifier.eval()

      running_loss = 0.0
      running_corrects = 0
      running_errors_ys = [0, 0, 0, 0]
      num_ys = [0, 0, 0, 0]
      ratio_errors_ys = [0, 0, 0, 0]
      ratio_upweight_ys = [0, 0, 0, 0]
      pred_ys = [0, 0, 0, 0]
      right_num = [0,0,0,0]
      loop = tqdm(dataloaders[mode], desc=mode)
      with (torch.enable_grad() if mode=="train" else torch.no_grad()):
        # cnt = 0
        for inputs, labels, sensitive_labels, names in loop:
          inputs = inputs.to(device)
          labels = labels.to(device)
          sensitive_labels = sensitive_labels.to(device)
          loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
          loop.set_postfix(loss = running_loss)

          with torch.no_grad():
             features = encoder(inputs)
             features = features.detach()
          outputs = classifier(features)
          loss = criterion(outputs, labels)
          # cnt += 1
          # print(cnt, loss.item())
          if mode == 'train':
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

          _, preds = torch.max(outputs, 1)
          running_loss += loss.item() * inputs.size(0)
          running_corrects += torch.sum(preds == labels.data)
          ys = labels*2+sensitive_labels
          ys_pred = preds*2+sensitive_labels
          mask_00 = (ys==0)
          mask_01 = (ys==1)
          mask_10 = (ys==2)
          mask_11 = (ys==3)
          
          num_error_00 = torch.sum(mask_00 * (preds != labels.data))
          num_error_01 = torch.sum(mask_01 * (preds != labels.data))
          num_error_10 = torch.sum(mask_10 * (preds != labels.data))
          num_error_11 = torch.sum(mask_11 * (preds != labels.data))
          running_errors_ys[0] += num_error_00.item()
          running_errors_ys[1] += num_error_01.item()
          running_errors_ys[2] += num_error_10.item()
          running_errors_ys[3] += num_error_11.item()
          num_00 = torch.sum(mask_00)
          num_01 = torch.sum(mask_01)
          num_10 = torch.sum(mask_10)
          num_11 = torch.sum(mask_11)
          num_ys[0] += num_00.item()
          num_ys[1] += num_01.item()
          num_ys[2] += num_10.item()
          num_ys[3] += num_11.item()

          pred_00 = (ys_pred==0)
          pred_01 = (ys_pred==1)
          pred_10 = (ys_pred==2)
          pred_11 = (ys_pred==3)
          num2_00 = torch.sum(pred_00)
          num2_01 = torch.sum(pred_01)
          num2_10 = torch.sum(pred_10)
          num2_11 = torch.sum(pred_11)
          pred_ys[0] += num2_00.item()
          pred_ys[1] += num2_01.item()
          pred_ys[2] += num2_10.item()
          pred_ys[3] += num2_11.item()

      
      epoch_loss = running_loss / len(image_datasets[mode])
      epoch_acc = running_corrects.double() / len(image_datasets[mode])
      
      print('{} loss: {:.4f}, acc: {:.4f}'.format(mode,
                                                  epoch_loss,
                                                  epoch_acc))
      print('num_errors_ys:', running_errors_ys)
      for i in range(4):
        ratio_errors_ys[i] = round(running_errors_ys[i] / num_ys[i], 2)
        right_num[i] = num_ys[i]-running_errors_ys[i]
        if sum(running_errors_ys)!=0:
          ratio_upweight_ys[i] = round(running_errors_ys[i]/sum(running_errors_ys), 2)
      acc,group_mean_acc,worst_acc,var,DP,EO = get_result(num_ys,pred_ys,right_num)
      print('ratio_errors_ys:', ratio_errors_ys)
      print('ratio_upweight_ys:', ratio_upweight_ys)
      print('pred_ys:', pred_ys)
      print('acc:',acc)
      print('group_mean_acc:',group_mean_acc)
      print('worst_acc:',worst_acc)
      print('var:',var)
      print('DP:',DP)
      print('EO:',EO)
      print("\n")
      if mode != "train":
          torch.save(classifier.state_dict(), args.output + '/model_epoch={}_acc={:02f}_worst={:02f}_EO={:02f}.pth'.format(epoch+1, acc, worst_acc, EO))
  return classifier

# def _convert_image_to_rgb(image):
#     return image.convert("RGB")

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument('--train', required=False,
                  help='path to folder train')
  ap.add_argument('--test', required=False, 
                  help='path to folder valid')
  ap.add_argument('--output', default='20000_0.111_degree=3,4,5-scale=0.8-135-epoch-trained_resnet18_BYOL_degree=4,5,6_0.0001_10000',          
                  help='path to save model trained')
  ap.add_argument('--y', default='Black_Hair',
                  help='name of target attribute')
  ap.add_argument('--s', default='Young',
                  help='name of sensitive attribute')
  args = ap.parse_args()

  args.output = args.y+"_"+args.s+"_"+args.output

  y = args.y
  s = args.s
  if not os.path.isdir(args.output):
      os.makedirs(args.output)

  image_datasets = {
      'train': 
      CelebA_degree(img_path=f"./data/two_degree_{args.y}_{args.s}_20000_0.111_0.0001_10000_different_degree_norm",attr_path=f"./data/list_attr_celeba_{args.y}_{args.s}_20000_0.111.csv",metadata_path=f'./data/metadata_{args.y}_{args.s}_20000_0.111.csv',mode="train", target=y, sensitive=s),

      'val':
      CelebA_degree(img_path=f"./data/img_align_celeba",attr_path=f"./data/list_attr_celeba_{args.y}_{args.s}_20000_0.111.csv",metadata_path=f'./data/metadata_{args.y}_{args.s}_20000_0.111.csv',mode="valid", target=y, sensitive=s),
     
  }

  dataloaders = {
      'train':
      torch.utils.data.DataLoader(image_datasets['train'],
                                  batch_size=128,
                                  shuffle=True,
                                  num_workers=12),
      'val':
      torch.utils.data.DataLoader(image_datasets['val'],
                                  batch_size=128,
                                  shuffle=False,
                                  num_workers=12),
      
  }
  
  dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
  print("Target Attribute:", y)
  print("Sensitive Attribute:", s)
  print(dataset_sizes)
  inputs, classes, _, _ = next(iter(dataloaders['train']))
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  encoder = BYOLResNet().cuda()
  model_path = './BYOL_encoder/BYOL_Smiling_Male_20000_0.111_degree=3,4,5_0.0001_10000_scale_0.8-models/BYOL_Smiling_Male_20000_0.111_degree=3,4,5_0.0001_10000_scale_0.8-model-feature-dim=128_batchsize=32_epoch=135.pth'
  pretrained_params = torch.load(model_path)
  encoder.load_state_dict(pretrained_params)
  print('Load trained model successfully!')

  print(model_path)

  classifier = LinearClassifier().cuda()
  optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)
  
  criterion = nn.CrossEntropyLoss()

  model_trained = train_model(encoder, classifier, criterion, optimizer, num_epochs=300)
  torch.save(model_trained.state_dict(), args.output + '/model_'+str(time.time() % 1000)+'.pth')
