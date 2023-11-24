## Adaptive Feature Aggregation Centric Enhance Network for Accurate and Fast Monocular 3D Object Detection
The respository is for AFACENet. The paper is submmit to IEEE Transactions on Instrumentation and Measurement. The paper is under review.

## Overall architecture
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/cosS32_K.jpeg)

## Detection result
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/S-nlcFMd.jpeg)

## Model link
[Our model](https://drive.google.com/file/d/1pE3A22MJeqjE_L8YuwoBWUqXRJ6hb9Qg/view?usp=sharing)

## Model training
./ddd_3dop.sh

or

python3 main.py --task ddd --exp_id AFACENet_no_s_new_center --arch dlapreasonnons_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 20 --num_epochs 100 --lr_step 45,60,70,80,90 --gpus 0 --num_workers 20

note: You can change the configuration in ddd_3dop.sh or the command

## Model demostration
./Monocular_3D_Detection_demo.sh

or

python3 ./src/demo.py --task ddd --arch dlapreasonnons_34 --load_model ./exp/ddd/AFACENet_no_s_new_center/model_100.pth --dataset kitti --demo single_image_image_folder_or_video

note: You can change the configuration in Monocular_3D_Detection_demo.sh or the command

## Model evaluation
./test

or

python3 test.py --task ddd --exp_id 3dop_dlap_reasonable_nc --arch dlapreasonnons_34 --dataset kitti --kitti_split 3dop --debug 4 --load_model ../exp/ddd/AFACENet_no_s_new_center/model_100.pth

note: You can change the configuration in test.sh or the command

## References
[1] [CenterNet](https://github.com/xingyizhou/CenterNet)

[2] [SFA3D](https://github.com/maudzung/SFA3D)

[3] [MobileViT](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilevit.py)

[4] [MobileFormer](https://github.com/kevinz8866/MobileFormer)

[5] [Vision Transformer with Deformable Attention](https://github.com/LeapLabTHU/DAT/tree/main)

[6] [VovNetv2](https://github.com/youngwanLEE/vovnet-detectron2)

[7] [ECANet](https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py?fbclid=IwAR305bvvHYF-q6SupbMvTtMkm0rAqMBjMCeIhC-HB6lFEPw5saEhqoIz3ZU)

[8] [Dynamic Convolution](https://github.com/kaijieshi7/Dynamic-convolution-Pytorch)

[9] [MobileNetv3](https://github.com/YaphetS-X/CenterNet-MobileNetV3)

[10] [CSPNet](https://zhuanlan.zhihu.com/p/263555330)

[11] [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR)
