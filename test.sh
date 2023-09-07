cd src
#python3 test.py --task ddd --exp_id 3dop_dla_dcnv1 --arch dla_34 --dataset kitti --kitti_split 3dop --load_model ../exp/ddd/3dop_dla_dcnv1/model_last.pth #--resume
#python3 test.py --task ddd --exp_id 3dop_dlap_non_transposeconv --arch dlapnontransposeconv_34 --dataset kitti --kitti_split 3dop --load_model ../exp/ddd/3dop_dlap_non_transposeconv/model_last.pth
#python3 test.py --task ddd --exp_id 3dop_dlap_non_carafe_full_new --arch dlapnoncarafefull_34 --dataset kitti --kitti_split 3dop --load_model ../exp/ddd/3dop_dlap_non_carafe_full_new/model_last.pth

python3 test.py --task ddd --exp_id 3dop_dlap_reasonable_v2_nc_visualization --arch dlapreasontwo_34 --dataset kitti --kitti_split 3dop --debug 4 --load_model ../exp/ddd/3dop_dlap_reasonable_v2_nc/model_70.pth

#python3 test.py --task ddd --exp_id 3dop_dlap_head_test --arch dlapheadtest_34 --dataset kitti --kitti_split 3dop --load_model ../exp/ddd/3dop_dlap_head_test/model_last.pth
cd ..
