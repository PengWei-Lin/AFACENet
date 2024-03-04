## Adaptive Feature Aggregation Centric Enhance Network for Accurate and Fast Monocular 3D Object Detection
The respository is for AFACENet. The paper is submmit to IEEE Transactions on Instrumentation and Measurement. The paper is under review.

## Overall architecture
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/LtimDmlS.jpeg)

## Detection result
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/QW3avUlX.jpeg)

## Comparison Detail
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/15add_pred.png)
### AFACENet
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/15bird_pred.png)
### AFACENet's BEV
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/16car_poser.png)
### RTM3D
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/16car_pose.png)
### KM3D
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/000031.png)
### SMOKE
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/16_output.jpg)
### LiDAR Ground Truth

![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/25add_pred.png)
### AFACENet
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/25bird_pred.png)
### AFACENet's BEV
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/26car_poser.png)
### RTM3D
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/26car_pose.png)
### KM3D
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/000052.png)
### SMOKE
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/26_output.jpg)
### LiDAR Ground Truth

![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/29add_pred.png)
### AFACENet
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/29bird_pred.png)
### AFACENet's BEV
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/30car_poser.png)
### RTM3D
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/30car_pose.png)
### KM3D
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/000061.png)
### SMOKE
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/30_output.jpg)
### LiDAR Ground Truth

![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/69add_pred.png)
### AFACENet
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/69bird_pred.png)
### AFACENet's BEV
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/70car_poser.png)
### RTM3D
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/70car_pose.png)
### KM3D
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/000167.png)
### SMOKE
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/70_output.jpg)
### LiDAR Ground Truth

![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/66add_pred.png)
### AFACENet
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/66bird_pred.png)
### AFACENet's BEV
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/67car_pose.png)
### RTM3D
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/67car_posekm.png)
### KM3D
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/000156.png)
### SMOKE
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/67_output.jpg)
### LiDAR Ground Truth

![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/72add_pred.png)
### AFACENet
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/72bird_pred.png)
### AFACENet's BEV
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/73car_poser.png)
### RTM3D
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/73car_pose.png)
### KM3D
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/000170.png.png)
### SMOKE
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/73_output.jpg)
### LiDAR Ground Truth

![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/74add_pred.png)
### AFACENet
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/74bird_pred.png)
### AFACENet's BEV
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/75car_poser.png)
### RTM3D
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/75car_pose.png)
### KM3D
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/000174.png.png)
### SMOKE
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/75_output.jpg)
### LiDAR Ground Truth


## Model link
[Our model](https://drive.google.com/file/d/19a2EZlV_THCz3UOiJo95KMKHdJkBrvq1/view?usp=sharing)

## Model training
./ddd_3dop.sh

or

python3 main.py --task ddd --exp_id AFANCENet_sc --arch dlasc_34 --head_conv 256 --dataset kitti --kitti_split 3dop --batch_size 20 --num_epochs 100 --lr_step 45,60,70,80,90 --gpus 0 --num_workers 20

note: You can change the configuration in ddd_3dop.sh or the command

## Model demostration
./Monocular_3D_Detection_demo.sh

or

python3 ./src/demo.py --task ddd --arch dlasc_34 --load_model ./exp/ddd/AFACENet_sc/model_100.pth --dataset kitti --demo single_image_image_folder_or_video

note: You can change the configuration in Monocular_3D_Detection_demo.sh or the command

## Model evaluation
./test

or

python3 test.py --task ddd --exp_id AFANCENet_sc --arch dlasc_34 --dataset kitti --kitti_split 3dop --debug 4 --load_model ../exp/ddd/AFANCENet_sc/model_80.pth

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
