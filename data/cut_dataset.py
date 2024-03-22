# -*- coding: utf-8 -*-

'''
Sampling from CelebA to construct a biased dataset.
'''

import shutil
import os
import pandas as pd
import pdb

#注意！！改attr_target_name，attr_sensitive_name，num_train_total，bias_degree

img_path = "./data/img_align_celeba/"
img_val_path = "./data/img_align_celeba/"
img_init_path = "./data/img_align_celeba/"
attr_path = "./data/list_attr_celeba.csv"
metadata_path = './data/metadata.csv'

attr_target_name = 'Smiling'
attr_sensitive_name = 'Male'
num_train_total = 20000
bias_degree = 1/9


# get image name
list_img_name_tr = []
list_img_name_val = []
list_img_name_te = []

# val set: 162771 - 182637
# test set: 182638 - 202599
for i in range(162771, 182638):
    # print(str(i).zfill(6)+".jpg")
    list_img_name_val.append(str(i).zfill(6)+".jpg")
for i in range(182638, 202600):
    # print(str(i).zfill(6)+".jpg")
    list_img_name_te.append(str(i).zfill(6)+".jpg")


df = pd.read_csv(attr_path)

ratio_ys = [bias_degree/(bias_degree+1)/2, 1/(bias_degree+1)/2, 1/(bias_degree+1)/2, bias_degree/(bias_degree+1)/2]
# ratio_ys = [1/(bias_degree+1)/2, bias_degree/(bias_degree+1)/2, bias_degree/(bias_degree+1)/2, 1/(bias_degree+1)/2]
print(ratio_ys)
num_train_ys_total = [round(ratio_ys[i]*num_train_total) for i in range(len(ratio_ys))]
print(num_train_ys_total)

count_train_ys = [0, 0, 0, 0]
for i in range(162770):
    if(os.path.exists(img_path+str(i+1).zfill(6)+".jpg")):
        y = int((df.loc[i][attr_target_name]+1)/2)
        s = int((df.loc[i][attr_sensitive_name]+1)/2)
        ys = y*2+s
        if count_train_ys[ys] < num_train_ys_total[ys]:
            list_img_name_tr.append(df.loc[i]['image_id'])
            count_train_ys[ys] += 1
        if count_train_ys == num_train_ys_total:
            print('count finished')
            break
        
if count_train_ys != num_train_ys_total:
    print(count_train_ys)
    print('error: training data is NOT enough')

    for i in range(162770):
        y = int((df.loc[i][attr_target_name]+1)/2)
        s = int((df.loc[i][attr_sensitive_name]+1)/2)
        ys = y*2+s
        # if yst==7:
        #     print(i,y,s,t,yst)
        if (count_train_ys[ys] < num_train_ys_total[ys]) & (df.loc[i]['image_id'] not in list_img_name_tr):
            list_img_name_tr.append(df.loc[i]['image_id'])
            count_train_ys[ys] += 1
        if count_train_ys == num_train_ys_total:
            print('count finished')
            break
    if count_train_ys != num_train_ys_total:
        print(count_train_ys)
        print('error: training data is NOT enough')
    


# move images
new_img_path = f"./data/jpg_align_celeba_{attr_target_name}_{attr_sensitive_name}_{num_train_total}_{bias_degree:.3f}_traindata/"
new_img_path_val = f"./data/val_full/"
new_img_path_test = f"./data/test/"

if not os.path.exists(new_img_path):
    os.makedirs(new_img_path)
if not os.path.exists(new_img_path_val):
    os.makedirs(new_img_path_val)
if not os.path.exists(new_img_path_test):
    os.makedirs(new_img_path_test)

print(len(list_img_name_tr))
for img_name in list_img_name_tr:
    if(os.path.exists(img_path+img_name[:6]+".jpg")):
        shutil.copyfile(img_path+img_name[:6]+".jpg", new_img_path+img_name[:6]+".jpg")
    else:
        shutil.copyfile(img_init_path+img_name[:6]+".jpg", new_img_path+img_name[:6]+".jpg")
# for img_name in list_img_name_val:
#     if(os.path.exists(img_val_path+img_name[:6]+".jpg")):
#         shutil.copyfile(img_val_path+img_name[:6]+".jpg", new_img_path_val+img_name[:6]+".jpg")
# for img_name in list_img_name_te:
#     if(os.path.exists(img_val_path+img_name[:6]+".jpg")):
#         shutil.copyfile(img_val_path+img_name[:6]+".jpg", new_img_path+img_name[:6]+".jpg")

if len(os.listdir(new_img_path)) == len(list_img_name_tr) + len(list_img_name_val) + len(list_img_name_te):
    print("#tr:", len(list_img_name_tr))
    print("#val:", len(list_img_name_val))
    print("#te:", len(list_img_name_te))
    print("image files are OK")
else:
    print("error: image files")


# generate csv
new_attr_path = f"./data/list_attr_celeba_{attr_target_name}_{attr_sensitive_name}_{num_train_total}_{bias_degree:.3f}.csv"
new_metadata_path = f"./data/metadata_{attr_target_name}_{attr_sensitive_name}_{num_train_total}_{bias_degree:.3f}.csv"

drop_index_list = []
for i in range(162770):
    # if(os.path.exists(img_path+str(i+1).zfill(6)+".jpg")):
    img_name_tmp = str(i+1).zfill(6)+".jpg"
    if (img_name_tmp not in list_img_name_tr) and (img_name_tmp not in list_img_name_val) and (img_name_tmp not in list_img_name_te):
        drop_index_list.append(i)
    # else:
    #     drop_index_list.append(i)

#print(drop_index_list)
df_new = df.drop(drop_index_list)

df_meta = pd.read_csv(metadata_path)
df_meta_new = df_meta.drop(drop_index_list)

# df.to_csv('try.csv',index=False)
# df.to_csv('try.csv',index=False)

df_new.to_csv(new_attr_path, index=False)
df_meta_new.to_csv(new_metadata_path, index=False)

