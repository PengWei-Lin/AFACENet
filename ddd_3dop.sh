cd src
# train
#python3 main.py --task ddd --exp_id 3dop_mobilenet --arch mobilenet --head_conv 64 --dataset kitti --kitti_split 3dop --batch_size 50 --master_batch 7 --num_epochs 2000 --lr_step 50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1100,1300,1500,1700,1850,1950 --gpus 0 --num_workers 14 --resume
#python3 main.py --task ddd --exp_id 3dop_dla_no_TransposeConv --arch dla_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 6 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6
#python3 main.py --task ddd --exp_id 3dop_mobilenet --arch mobilenetv3s --head_conv 64 --dataset kitti --kitti_split 3dop --batch_size 16 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6
#python3 main.py --task ddd --exp_id 3dop_mobilenetv1 --arch mobilenetv1 --head_conv 64 --dataset kitti --kitti_split 3dop --batch_size 6 --master_batch 7 --num_epochs 10 --lr_step 5 --gpus 0 --num_workers 6
#python3 main.py --task ddd --exp_id 3dop_mobilenet --arch mobilenetv3csp --head_conv 64 --dataset kitti --kitti_split 3dop --batch_size 8 --master_batch 7 --num_epochs 300 --lr_step 10,20,50,70,100,120,150,170,200,220,250,270,290 --gpus 0 --num_workers 6

#python3 main.py --task ddd --exp_id 3dop_mobilenet --arch mobilenetv3 --head_conv 64 --dataset kitti --kitti_split 3dop --batch_size 8 --master_batch 7 --num_epochs 300 --lr_step 10,20,50,70,100,120,150,170,200,220,250,270,290 --gpus 0 --num_workers 6 #--resume



#python3 main.py --task ddd --exp_id 3dop_hrnet18 --arch hrnet18 --head_conv 64 --dataset kitti --kitti_split 3dop --batch_size 5 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6



#python3 main.py --task ddd --exp_id 3dop_dla_no_TransposeConv --arch dla_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 5 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6


#python3 main.py --task ddd --exp_id 3dop_dla --arch dla_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 6 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6

#python3 main.py --task ddd --exp_id 3dop_dlav2 --arch dla2v_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 5 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6 --resume #ResBlock


#python3 main.py --task ddd --exp_id 3dop_dlav2 --arch dla2v_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 5 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6 --resume



#python3 main.py --task ddd --exp_id 3dop_dla_200epoch --arch dla_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 16 --num_epochs 200 --lr_step 45,60,75,100,145,160,190 --gpus 0 --num_workers 20 #--master_batch 7 --num_workers 6    # max batch size 22



#python3 main.py --task ddd --exp_id 3dop_dlap_reasonable --arch dlapreason_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 20 --num_epochs 100 --lr_step 45,60,70,80,90 --gpus 0 --num_workers 20


#python3 main.py --task ddd --exp_id 3dop_dlap_reasonable_branch --arch dlapreasonbranch_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 20 --num_epochs 100 --lr_step 45,60,70,80,90 --gpus 0 --num_workers 20


#python3 main.py --task ddd --exp_id 3dop_dlap_reasonable_nc --arch dlapreason_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 20 --num_epochs 100 --lr_step 45,60,70,80,90 --gpus 0 --num_workers 20

#python3 main.py --task ddd --exp_id 3dop_dlap_reasonable_v2_nc --arch dlapreasontwo_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 20 --num_epochs 100 --lr_step 45,60,70,80,90 --gpus 0 --num_workers 20


#python3 main.py --task ddd --exp_id 3dop_dlap_reasonable_branch_only --arch dlapreasonbranchonly_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 20 --num_epochs 100 --lr_step 45,60,70,80,90 --gpus 0 --num_workers 20


#python3 main.py --task ddd --exp_id 3dop_dlap_final_KFPN --arch dlapfinalkfpn_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 20 --num_epochs 100 --lr_step 45,60,70,80,90 --gpus 0 --num_workers 20


python3 main.py --task ddd --exp_id 3dop_dlap_head_test --arch dlapheadtest_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 20 --num_epochs 100 --lr_step 45,60,70,80,90 --gpus 0 --num_workers 20


#python3 main.py --task ddd --exp_id 3dop_dlap_final_KFPN_v2 --arch dlapfinalkfpntwo_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 20 --num_epochs 100 --lr_step 45,60,70,80,90 --gpus 0 --num_workers 20


#python3 main.py --task ddd --exp_id 3dop_dlap_head_test_v2 --arch dlapheadtesttwo_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 20 --num_epochs 100 --lr_step 45,60,70,80,90 --gpus 0 --num_workers 20


#python3 main.py --task ddd --exp_id 3dop_dlap_non_transposeconv --arch dlapnontransposeconv_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 20 --num_epochs 100 --lr_step 45,60,70,80,90 --gpus 0 --num_workers 20



#python3 main.py --task ddd --exp_id 3dop_dlap --arch dlap_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 5 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6



#python3 main.py --task ddd --exp_id 3dop_mvitdla --arch mvitdla_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 4 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6 --resume



#python3 main.py --task ddd --exp_id 3dop_mfdla --arch mfdla_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 3 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6



#python3 main.py --task ddd --exp_id 3dop_ldla --arch ldla_34 --head_conv 128 --dataset kitti --kitti_split 3dop --batch_size 6 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6 *****************Fail


#python3 main.py --task ddd --exp_id 3dop_ldla --arch dla_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 5 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6

#python3 main.py --task ddd --exp_id 3dop_dla_multiv2 --arch dla_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 4 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6

#python3 main.py --task ddd --exp_id 3dop_MobileFormer52 --arch mobileformer52 --head_conv 64 --dataset kitti --kitti_split 3dop --batch_size 10 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6

#python3 main.py --task ddd --exp_id 3dop_MobileFormer96 --arch mobileformer96 --head_conv 64 --dataset kitti --kitti_split 3dop --batch_size 10 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6

#python3 main.py --task ddd --exp_id 3dop_MobileFormer151 --arch mobileformer151 --head_conv 64 --dataset kitti --kitti_split 3dop --batch_size 8 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6


#python3 main.py --task ddd --exp_id 3dop_MobileFormer294 --arch mobileformer294 --head_conv 64 --dataset kitti --kitti_split 3dop --batch_size 6 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6

#python3 main.py --task ddd --exp_id 3dop_MobileFormer508 --arch mobileformer508 --head_conv 64 --dataset kitti --kitti_split 3dop --batch_size 4 --master_batch 7 --num_epochs 200 --lr_step 20,40,60,80,100,120,140,160,180,190 --gpus 0 --num_workers 6

#python3 main.py --task ddd --exp_id 3dop_MobileViTxxs --arch mobilevitxxs --head_conv 64 --dataset kitti --kitti_split 3dop --batch_size 2 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6

#python3 main.py --task ddd --exp_id 3dop_MobileViTxs --arch mobilevitxs --head_conv 64 --dataset kitti --kitti_split 3dop --batch_size 2 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6

#python3 main.py --task ddd --exp_id 3dop_MobileViTs --arch mobilevits --head_conv 64 --dataset kitti --kitti_split 3dop --batch_size 2 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6


#python3 main.py --task ddd --exp_id 3dop_Deformable_Attention --arch dat --head_conv 192 --dataset kitti --kitti_split 3dop --batch_size 1 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6


#python3 main.py --task ddd --exp_id 3dop_mobilevitv2 --arch mobilevitv2 --head_conv 128 --dataset kitti --kitti_split 3dop --batch_size 4 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6

#python3 main.py --task ddd --exp_id 3dop_mobilevitv3xxs --arch mobilevitv3xxs --head_conv 64 --dataset kitti --kitti_split 3dop --batch_size 2 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6


#python3 main.py --task ddd --exp_id 3dop_mobilevitv3xxst --arch mobilevitv3xxst --head_conv 64 --dataset kitti --kitti_split 3dop --batch_size 2 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6

#python3 main.py --task ddd --exp_id 3dop_mobilevitv3xxstt --arch mobilevitv3xxstt --head_conv 64 --dataset kitti --kitti_split 3dop --batch_size 2 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6



#python3 main.py --task ddd --exp_id 3dop_MobileFormer52ttt --arch mobileformer52ttt --head_conv 64 --dataset kitti --kitti_split 3dop --batch_size 2 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6



#python3 main.py --task ddd --exp_id 3dop_MobileFormer52tt --arch mobileformer52tt --head_conv 64 --dataset kitti --kitti_split 3dop --batch_size 10 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6 --resume


#python3 main.py --task ddd --exp_id 3dop_MobileFormer52t --arch mobileformer52t --head_conv 12 --dataset kitti --kitti_split 3dop --batch_size 22 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6



#python3 main.py --task ddd --exp_id 3dop_mobilevitv3d1d0 --arch mobilevitv3d1d0 --head_conv 128 --dataset kitti --kitti_split 3dop --batch_size 3 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6 --resume


#python3 main.py --task ddd --exp_id 3dop_mobilevitv3d1d0t --arch mobilevitv3d1d0t --head_conv 128 --dataset kitti --kitti_split 3dop --batch_size 3 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6



#python3 main.py --task ddd --exp_id 3dop_dla_no_TransposeConv_loss_test --arch dla_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 5 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6

#python3 main.py --task ddd --exp_id 3dop_resnet_no_TransposeConv --arch resdcn_50 --head_conv 64 --dataset kitti --kitti_split 3dop --batch_size 5 --master_batch 7 --num_epochs 100 --lr_step 10,20,30,40,50,60,70,80,90,95 --gpus 0 --num_workers 6

# test
#python3 test.py --task ddd --exp_id 3dop --dataset kitti --kitti_split 3dop --resume
cd ..
