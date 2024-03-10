choose=0.9
name=Smiling_Male_20000_0.111
for i in {3,4,5,6,-3,-4,-5,-6}
do
        python ./scripts/inference.py --images_dir=./test_imgs  --n_sample=200000 --edit_attribute='smile' --edit_degree=$i \
        --save_dir="./data/two_degree_"$name"_0.001_1000_different_degree_norm" \
        --directions_path1="./editings/celeba_directions/"$name"_linear_0.0001_10000_transfer_$choose.pth"  \
        --directions_path2="./editings/celeba_directions/"$name"_linear_0.0001_10000_transfer_$choose.pth"  \
        --data_dir="./data/jpg_align_celeba_"$name"_traindata" \
        --attr="/data/list_attr_celeba_$name.csv"  \
        --meta="/data/metadata_$name.csv"  \
        --t="Smiling"  --s="Male"  \
        ./experiment/Smiling_Male_20000_0.111_align/checkpoints/iteration_150000.pt
    wait $!
done

python ./BYOL/BYOL.py --y Smiling --s Male --name Smiling_Male_20000_0.111

python ./BYOL/train_LC.py --y Smiling --s Male

