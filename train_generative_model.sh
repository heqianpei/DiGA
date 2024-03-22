python ./scripts/train.py   --dataset_type='ffhq_encode'  --start_from_latent_avg \
--id_lambda=0.1  --val_interval=5000 --save_interval=5000 --max_steps=210000  --stylegan_size=1024 --is_train=True \
--distortion_scale=0.15 --aug_rate=0.9 --res_lambda=0.1  \
--stylegan_weights='' --checkpoint_path=''  \
--workers=12  --batch_size=4  --test_batch_size=4 --test_workers=12 --exp_dir='./experiment/Smiling_Male_20000_0.111_align'