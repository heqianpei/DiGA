
python get_latent_codes.py --data_path "./data/jpg_align_celeba_Smiling_Male_20000_0.111_traindata/" \
--model_path "./experiment/Smiling_Male_20000_0.111_align/checkpoints/iteration_150000.pt"  \
--result_path "./data/align_latent_code_Smiling_Male_20000_0.111_traindata/"

python /home/zhangfengda/DiGA/boundary/train_boundary.py

name=Smiling_Male_20000_0.111
choose1_list=(10000)
choose2_list=(0.0001)

for ((index=0; index<${#choose1_list[@]}; index++))
do
    choose1=${choose1_list[index]}
    for ((indexx=0; indexx<${#choose2_list[@]}; indexx++))
    do
        choose2=${choose2_list[indexx]}
        for i in $(seq 0.5 0.1 1.5)
        do
            python get_transfer.py --degree $i --name $name --choose1 $choose1 --choose2 $choose2
            wait $!
            for j in {5,-5}
            do
                CUDA_VISIBLE_DEVICES='1' setsid python -u ./scripts/inference_test.py --images_dir=./test_imgs  --n_sample=100 --edit_attribute='smile' --edit_degree=$j \
                    --save_dir="./new_image/custom_results_"$name"_"$choose2"_"$choose1"/degree_$i" \
                    --directions_path1="/home/zhangfengda/HFGI/editings/celeba_directions/"$name"_linear_"$choose2"_"$choose1"_transfer_$i.pth"  \
                    --data_dir="/home/zhangfengda/CelebA/jpg_align_celeba_"$name"_traindata" \
                    --attr="/home/zhangfengda/CelebA/list_attr_celeba_$name.csv"  \
                    --meta="/home/zhangfengda/CelebA/metadata_$name.csv"  \
                    --t="Smiling"  --s="Male"  \
                    /home/zhangfengda/HFGI/experiment/"$name"_align/checkpoints/iteration_150000.pt  > train_log_$name.txt
                wait $!
            done
        done
    done
done