## Adaptive Feature Aggregation Centric Enhance Network for Accurate and Fast Monocular 3D Object Detection
The respository is for AFACENet. The paper is submmit to IEEE Transections on Instrument and Meansurement. The paper is under review. The repository is still in progress.

## Overall architecture
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/377734717_3568350886770723_955616644508864195_n.png)

## Detection result
![image](https://github.com/PengWei-Lin/AFACENet/blob/main/pic/377268711_2331490857034091_4453226347110264732_n.png)

## Model link
[Our model](https://drive.google.com/file/d/1zUgyva-F8SX_YwBInAypGsnGioDbmLKS/view?usp=sharing)

## Model training
./ddd_3dop.sh
note: you can change the configuration in ddd_3dop.sh

## Model demostration
./Monocular_3D_Detection_demo.sh
note: you can change the configuration in Monocular_3D_Detection_demo.sh

## Model evaluation
./test
note: you can change the configuration in test.sh

## References
[1] [CenterNet](https://github.com/xingyizhou/CenterNet)

[2] [SFA3D](https://github.com/maudzung/SFA3D)

[3] [MobileViT](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilevit.py)

[4] [MobileFormer](https://github.com/kevinz8866/MobileFormer)

[5] [Vision Transformer with Deformable Attention](https://github.com/LeapLabTHU/DAT/tree/main)
