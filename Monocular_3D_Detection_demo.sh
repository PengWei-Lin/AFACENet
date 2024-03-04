#python3 ./src/demo.py tracking,ddd --load_model ./models/own_3Dtracking.pth --demo ./videos/24_2022-05-11.mp4 --test_focal_length 622 --num_classes 2 --dataset nuscenes --input_h 480 --input_w 640 --video_h 480 --video_w 640 --save_video
#python3 ./src/demo.py tracking,ddd --load_model ./models/nuScenes_3Dtracking.pth --demo ./videos/nuscenes_mini.mp4 --dataset nuscenes --save_video
#python3 ./src/demo.py tracking,ddd --load_model ./models/nuScenes_3Dtracking.pth --demo ./videos/24_2022-05-11.mp4 --test_focal_length 622 --dataset nuscenes --input_h 480 --input_w 640 --video_h 480 --video_w 640 --save_video

#python3 ./src/demo.py --task ddd --arch dla_34 --load_model ./exp/ddd/3dop_dla/model_best.pth --demo ./videos/output.mp4 --dataset kitti #--input_h 480 --input_w 640
#python3 ./src/demo.py --task ddd --arch dla_34 --load_model ./exp/ddd/3dop_dla/model_best.pth --demo ./videos/24_2022-05-11.mp4 --dataset kitti --input_h 480 --input_w 640
#python3 ./src/demo.py --task ddd --arch mvitdla_34 --load_model ./exp/ddd/3dop_mvitdla/best_2_7986_with_dla_pretrianed_model_and_CSAMV2_around_15_epoch/model_best.pth --demo ./videos/output.mp4 --dataset kitti
#python3 ./src/demo.py --task ddd --arch mvitdla_34 --load_model ./exp/ddd/3dop_mvitdla/model_best.pth --demo ./videos/24_2022-05-11.mp4 --dataset kitti

python3 ./src/demo.py --task ddd --arch dlasc_34 --load_model ./exp/ddd/AFACENet_sc/model_100.pth --dataset kitti --demo single_image_image_folder_or_video

#python3 ./src/demo.py --task ddd --arch dla2v_34 --load_model ./exp/ddd/3dop_dlav2/model_best.pth --demo ./videos/24_2022-05-11.mp4 --dataset kitti --input_h 480 --input_w 640


#python3 ./src/demo.py --task ddd --arch dlap_34 --load_model ./exp/ddd/3dop_dlap/model_best.pth --demo ./videos/24_2022-05-11.mp4 --dataset kitti --input_h 480 --input_w 640


#python3 ./src/demo.py --task ddd --arch mobileformer508 --load_model ./models/ddd_3dop_MobileFormer508.pth --demo ./videos/24_2022-05-11.mp4 --dataset kitti --input_h 480 --input_w 640
