# training
python train.py --save_path ./ckpt/issue --training_mode full_shot

# testing
declare -a dataset=(MVTec BTAD MPDD Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir Colon_cvc300)
save_path="./ckpt/issue"
for i in "${dataset[@]}"; do
    python test.py --save_path $save_path --dataset $i
done