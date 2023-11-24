cd src


python3 main.py --task ddd --exp_id AFACENet_no_s_new_center --arch dlapreasonnons_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 20 --num_epochs 150 --lr_step 45,60,70,80,90,100,110,120,130,140 --gpus 0 --num_workers 20  

cd ..


