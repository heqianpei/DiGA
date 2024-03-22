dataset_paths = {
	'ffhq_val': './data/val_HFGI',
	'ffhq': './data/jpg_align_celeba_Smiling_Male_20000_0.111_traindata',

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
