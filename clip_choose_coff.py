import os
import torch
import clip
from PIL import Image
from celeba import CelebA
import numpy as np
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

text = clip.tokenize(["a face without smile", "a face with smile"]).to(device)
text_features = model.encode_text(text)
image_path = "./new_image/custom_results_Smiling_Male_20000_0.111_0.0001_10000/"
subfolders = [f.path for f in os.scandir(image_path) if f.is_dir()]
subfolders = sorted(subfolders)
criterion = nn.CrossEntropyLoss()
with torch.no_grad():
    max_degree = 0
    result = ""
    for subfolder in subfolders:
        image_datasets = CelebA(img_path=subfolder,
                    attr_path=f"./data/attr_Smiling_Male_20000_0.111.csv",
                    metadata_path=f'./data/metadata_Smiling_Male_20000_0.111.csv',
                    image_size=224, mode="train",
                    target="Smiling", sensitive="Male")
        dataloaders = torch.utils.data.DataLoader(image_datasets,
                                    batch_size=1,
                                    shuffle=True,
                                    num_workers=12)
        result_acc = 0
        running_loss = 0
        running_errors_ys = [0, 0]
        num_ys = [0, 0]
        pred_ys = [0, 0]
        for image, labels in dataloaders:
            # image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
            # import pdb
            # pdb.set_trace()
            image = image[0].unsqueeze(0).to(device)
            # import pdb
            # pdb.set_trace()
            image_features = model.encode_image(image)
            logits_per_image, logits_per_text = model(image, text)
            loss = criterion(logits_per_image.to(device), labels.to(device))
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            preds = np.argmax(probs[0])
            
            
            # _, preds = torch.max(probs, 1)
            # mask_00 = (ys==0)
            # mask_01 = (ys==1)
            
            # num_error_00 = torch.sum(mask_00 * (preds != labels.data))
            # num_error_01 = torch.sum(mask_01 * (preds != labels.data))
            # running_errors_ys[0] += num_error_00.item()
            # running_errors_ys[1] += num_error_01.item()
            # num_00 = torch.sum(mask_00)
            # num_01 = torch.sum(mask_01)

            # num_ys[0] += num_00.item()
            # num_ys[1] += num_01.item()

            # pred_00 = (preds==0)
            # pred_01 = (preds==1)

            # num2_00 = torch.sum(pred_00)
            # num2_01 = torch.sum(pred_01)

            # pred_ys[0] += num2_00.item()
            # pred_ys[1] += num2_01.item()

        # for i in range(2):
        #     result_acc += (num_ys[i]-running_errors_ys[i])
            if(preds==labels):
                result_acc += 1
            running_loss += loss.item() * image.size(0)
        result_acc = result_acc/200
        running_loss = running_loss*2
        # print(subfolder, result_acc,running_loss)
        # if(result_acc>max_degree):
        #     max_degree = result_acc
        #     result = subfolder

        print(subfolder," acc: ",result_acc)