dataset_paths = {
	#  Face Datasets (FFHQ - train, CelebA-HQ - test)
	# 'ffhq': '/home/zhangfengda/CelebA/jpg_align_celeba_Smiling_Young_20000_0.111_traindata',
	# 'ffhq': '/home/zhangfengda/CelebA/jpg_align_celeba_Blond_Hair_Male_20000_0.111_traindata',
    # 'ffhq': '/home/zhangfengda/CelebA/jpg_align_celeba_Smiling_Male_Young_20000_1:2:2:15_traindata',
    # 'ffhq': '/home/zhangfengda/CelebA/jpg_align_celeba_Big_Nose_Male_20000_0.111_traindata',
	'ffhq_val': '/home/zhangfengda/CelebA/val_HFGI',
	'ffhq': '/home/zhangfengda/CelebA/jpg_align_celeba_Smiling_Male_20000_0.111_traindata',
	# 'ffhq_val': '/home/zhangfengda/UTKface/HFGI_test',

	#  Cars Dataset (Stanford cars)
	'cars_train': '',
	'cars_val': '',
    
	'dog_and_cat_train':'/home/zhangfengda/dogs-vs-cats/dogs_vs_cats_9000_0.125_v2',
    'dog_and_cat_val':'/home/zhangfengda/dogs-vs-cats/test2',
}

model_paths = {
	# 'stylegan_ffhq': './pretrained/stylegan2-ffhq-config-f.pt',
	# 'stylegan_ffhq': '/home/zhangfengda/stylegan2-pytorch-master/checkpoint/150000.pt',
    
	# 'ir_se50': './pretrained/model_ir_se50.pth',
	# 'ir_se50': '/home/zhangfengda/encoder4editing-main/newA100/checkpoints/iteration_150000.pt',
	'shape_predictor': './pretrained/shape_predictor_68_face_landmarks.dat',
	'moco': '/home/zhangfengda/encoder4editing-main/pretrained_models/moco_v2_800ep_pretrain.pth'
}
