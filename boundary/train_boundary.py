import tempfile
import numpy as np
from argparse import Namespace
from pathlib import Path
import torch
from torchvision import transforms
import PIL.Image
import scipy
import scipy.ndimage
import numpy as np
from sklearn import svm,linear_model
# from boundary.celeba_latent import CelebA
# from boundary.celeba_latent import CelebA_selection
from celeba_latent import CelebA_latent
from tqdm import tqdm
import os
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import joblib
import numpy as np
import argparse




def train_boundary(C_num,attr_pa,meta_pa,tar,sen,name,output,datasets,mode):
    # output = "/home/zhangfengda/interfacegan-master/boundary_result/unbias_smiling"
    unbias = "/home/zhangfengda/HFGI/editings/custom_directions/unbias_smiling.pth"
    unbias2 = "/home/zhangfengda/HFGI/editings/custom_directions/unbias_male.pth"
    if not os.path.isdir(output):
        os.makedirs(output)

    if mode==0:
        image_datasets = {
            'train': 
            CelebA_latent(img_path=datasets, attr_path=attr_pa,
                    metadata_path=meta_pa, image_size=256, mode="train", target=tar, sensitive=sen),
            # 'val': 
            # CelebA(img_path="/home/zhangfengda/CelebA/val", attr_path="/home/heqianpei/CelebA/list_attr_celeba_y=Smiling_s=Male_size=20000_bias=0.25.csv",
            #       metadata_path='/home/heqianpei/CelebA/metadata_y=Smiling_s=Male_size=20000_bias=0.25.csv', image_size=256, mode="valid", target="Smiling", sensitive="Male")
            #   'test': 
            #   CelebA(img_path=f"/home/heqianpei/CelebA/img_align_celeba_y={y}_s={s}_size=20000_bias=0.25/", attr_path=f"/home/heqianpei/CelebA/list_attr_celeba_y={y}_s={s}_size=20000_bias=0.25.csv",
            #         metadata_path=f'/home/heqianpei/CelebA/metadata_y={y}_s={s}_size=20000_bias=0.25.csv', image_size=128, mode="test", target=y, sensitive=s)
        }
    # elif mode==1:
    #     image_datasets = {
    #         'train': 
    #         CelebA_latent_selection(img_path=datasets, attr_path=f"/home/zhangfengda/CelebA/list_attr_celeba_y=Blond_Hair_s=Male_size=20000_bias=0.05263157894736842.csv",
    #                 metadata_path=f'/home/zhangfengda/CelebA/metadata_y=Blond_Hair_s=Male_size=20000_bias=0.05263157894736842.csv', image_size=256, mode="train", target="Smiling", sensitive="Male"),
    #         }
        
    dataloaders = {
        'train':
        torch.utils.data.DataLoader(image_datasets['train'],
                                    # batch_size=len(image_datasets["train"]),
                                    batch_size=20000,
                                    shuffle=True,
                                    num_workers=0),
        # 'val':
        # torch.utils.data.DataLoader(image_datasets['val'],
        #                             batch_size=64,
        #                             shuffle=False,
        #                             num_workers=0),
        #   'test':
        #   torch.utils.data.DataLoader(image_datasets['test'],
        #                               batch_size=64,
        #                               shuffle=False,
        #                               num_workers=12)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
    print(dataset_sizes)
    inputs, classes, _, _ = next(iter(dataloaders['train']))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = svm.SVC(kernel='linear')
    model = linear_model.LogisticRegression(C=C_num)
    # model = model.to(device)
    # model.cuda()

    
    # Observe that all parameters are being optimized
    #optimizer_ft = optim.Adam(model.parameters(), lr=0.001)
    #optimizer_ft = optim.Adagrad(model_ft.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochsScratch
    # Final Thoughts and Where to Go Next
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    for i, (images, labels,_,_) in enumerate(dataloaders["train"]):
        # import pdb
        # pdb.set_trace()
        print(i)
    # 将图像数据展平为一维数组
        # images = images.view(images.size(0), -1)
    # 训练SVM模型
        inputs = images.cpu().detach().numpy().squeeze()
        new_inputs = np.zeros((len(inputs),512))
        for j in range(0,len(inputs)):
            new_inputs[j] = np.average(inputs[j],axis=0)
        images = new_inputs
        input_labels = labels.numpy()
        model_trained = model.fit(images, input_labels)
        break
    a = model_trained.coef_.reshape(1, 512).astype(np.float32)

    #     inputs = images.cpu().detach().numpy().squeeze()
    #     new_inputs = np.zeros((len(inputs),512*14))
    #     for j in range(0,len(inputs)):
    #         new_inputs[j] = inputs[j].reshape(-1)
    #     images = new_inputs
    #     input_labels = labels.numpy()
    #     model_trained = model.fit(images, input_labels)

    # a = model_trained.coef_.reshape(1, 512*14).astype(np.float32)

    temp_a = a / np.linalg.norm(a)
    a = torch.from_numpy(temp_a)
    a = torch.nn.functional.normalize(a,dim = -1)
    print(a.size)
    torch.save(a, output+"/"+name+".pth")
    joblib.dump(model_trained,output+"/"+name+".pkl",compress=3)

if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser(description="Inference")
    # parser.add_argument("--degree", type=float, default=None, help="The directory to the images")

    args = parser.parse_args()
    train_boundary(0.0001,"./data/list_attr_celeba_Smiling_Male_20000_0.111.csv","./data/metadata_Smiling_Male_20000_0.111.csv","Smiling","Male","Smiling_Male_20000_0.111_linear_0.0001","./editings/celeba_directions","./data/align_latent_code_Smiling_Male_20000_0.111_traindata",0)
    train_boundary(10000,"./data/list_attr_celeba_Smiling_Male_20000_0.111.csv","./data/metadata_Smiling_Male_20000_0.111.csv","Smiling","Male","Smiling_Male_20000_0.111_linear_10000","./editings/celeba_directions","./data/align_latent_code_Smiling_Male_20000_0.111_traindata",0)

