cd src


python3 main.py --task ddd --exp_id AFANCENet_sc --arch dlasc_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 20 --num_epochs 100 --lr_step 45,60,70,80,90 --gpus 0 --num_workers 20

cd ..


