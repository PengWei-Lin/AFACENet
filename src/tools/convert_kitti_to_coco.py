from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np
import cv2
DATA_PATH = "../../data/kitti/"
DEBUG = False
# VAL_PATH = DATA_PATH + 'training/label_val/'
import os
SPLITS = ['3dop', 'subcnn'] 
import _init_paths
from utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d

'''
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''


################################################### For nuscenes
def _rot_y2alpha(rot_y, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    alpha = rot_y - np.arctan2(x - cx, fx)
    if alpha > np.pi:
      alpha -= 2 * np.pi
    if alpha < -np.pi:
      alpha += 2 * np.pi
    return alpha
##################################################



def _bbox_to_coco_bbox(bbox):
  return [(bbox[0]), (bbox[1]),
          (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]

def read_clib(calib_path):
  f = open(calib_path, 'r')
  for i, line in enumerate(f):
    if i == 2:
      calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
      calib = calib.reshape(3, 4)
      return calib
  
# Add amodel center    
def read_clib_rect_Tr(calib_path):
  f = open(calib_path, 'r')
  for i, line in enumerate(f):
    if i == 2:
      calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
      calib = calib.reshape(3, 4)
    elif i == 4:
      R0 = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
      R0 = R0.reshape(3, 3)
    elif i == 5:  #6 not sure
      Tr = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
      Tr = Tr.reshape(3, 4)
      #print(Tr)
  return calib, R0, Tr
# Add amodel center

#cats = ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck',  'Person_sitting',
        #'Tram', 'Misc', 'DontCare']
#cats = ['Car']
#cats = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 
                   #'pedestrian', 'motorcycle', 'bicycle',
                   #'traffic_cone', 'barrier']
cats = ['pedestrian','car', 'bicycle','truck', 'bus', 'trailer', 'construction_vehicle', 
        'motorcycle', 'traffic_cone', 'barrier']
cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}
# cat_info = [{"name": "pedestrian", "id": 1}, {"name": "vehicle", "id": 2}]
#F = 721
F = 1266
#F = 1472
#F = 2077
#F = 707
#H = 384 # 375
#H = 1200
#H = 1280
H = 900
#H = 720
#W = 1248 # 1242
W = 1600
#W = 1920
#W = 1280
#EXT = [45.75, -0.34, 0.005]
EXT = [0, 0, 0]
CALIB = np.array([[F, 0, W / 2, EXT[0]], [0, F, H / 2, EXT[1]], 
                  [0, 0, 1, EXT[2]]], dtype=np.float32)

cat_info = []
for i, cat in enumerate(cats):
  cat_info.append({'name': cat, 'id': i + 1})

for SPLIT in SPLITS:
  #image_set_path = DATA_PATH + 'ImageSets_{}/'.format(SPLIT)
  image_set_path = '../../data/kitti/ImageSets_{}/'.format(SPLIT)
  #ann_dir = DATA_PATH + 'training/label_2/'
  ann_dir = '../../data/kitti/training/label_2/'
  #calib_dir = DATA_PATH + '{}/calib/'
  calib_dir = '../../data/kitti/{}/calib/'
  splits = ['train', 'val']
  # splits = ['trainval', 'test']
  calib_type = {'train': 'training', 'val': 'training', 'trainval': 'training',
                'test': 'testing'}
  
  for split in splits:
    ret = {'images': [], 'annotations': [], "categories": cat_info}
    image_set = open(image_set_path + '{}.txt'.format(split), 'r')
    image_to_id = {}
    for line in image_set:
      if line[-1] == '\n':
        line = line[:-1]
      image_id = int(line)
      calib_path = calib_dir.format(calib_type[split]) + '{}.txt'.format(line)
      #calib = read_clib(calib_path)
      
      calib, R0, Tr = read_clib_rect_Tr(calib_path)
      
      calib_tl = calib.tolist()
      R0_tl = R0.tolist()
      Tr_tl = Tr.tolist()
      
      image_info = {'file_name': '{}.png'.format(line),
                    'id': int(image_id),
                    'calib': calib_tl,#calib.tolist,
                    'R0': R0_tl,#R0.tolist(),
                    'Tr': Tr_tl#Tr.tolist()
                    }
      ret['images'].append(image_info)
      if split == 'test':
        continue
      ann_path = ann_dir + '{}.txt'.format(line)
      # if split == 'val':
      #   os.system('cp {} {}/'.format(ann_path, VAL_PATH))
      
      P2 = np.array(calib_tl, dtype=np.float32)
      R0 = np.array(R0_tl, dtype=np.float32)
      Tr = np.array(Tr_tl, dtype=np.float32)
      
      new_r = [0, 0, 0, 0]
      P2_n = np.vstack((P2, new_r))
      
      new_c = [0, 0, 0]
      new_nr = [0, 0, 0, 1]
      R0_n = np.column_stack((R0, new_c))
      R0_n = np.vstack((R0_n, new_nr))
      Tr_n = np.vstack((Tr, new_nr))
      
      R0_inv = np.linalg.inv(R0_n)
      Tr_inv = np.linalg.inv(Tr_n)
      #Tr_inv = np.linalg.pinv(Tr_n)
      
      
      
      anns = open(ann_path, 'r')
      
      if DEBUG:
        image = cv2.imread(
          DATA_PATH + 'images/trainval/' + image_info['file_name'])
        #print(DATA_PATH + 'images/trainval/' + image_info['file_name'])
          #DATA_PATH + 'training/image_2/' + image_info['file_name'])

      for ann_ind, txt in enumerate(anns):
        tmp = txt[:-1].split(' ')
        cat_id = cat_ids[tmp[0]]
        truncated = int(float(tmp[1]))
        occluded = int(tmp[2])
        ############################# kitti
        #alpha = float(tmp[3])
        #############################
        bbox = [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])]
        dim = [float(tmp[8]), float(tmp[9]), float(tmp[10])]
        location = [float(tmp[11]), float(tmp[12]), float(tmp[13])]
        rotation_y = float(tmp[14])
        
        # Add amodel center
        # System 1
        '''
        x,y,z = location
        
        h,w,l = dim
        
        pts_3d = [x, y-(h/2), z, 1]
        pts_3d = np.matmul(R0_n, pts_3d)
        pts_2d = np.dot(P2_n, pts_3d)
        pts_2d = pts_2d[:2] / pts_2d[2:3]
        x, y = pts_2d
        nc_xy = [x, y]
        '''
        
        
        # System 2
        xn, yn, zn = location
        hn, wn, ln = dim
        pts_3dn = [xn, yn-(hn/2), zn, 1]
        pts_3dn = np.matmul(R0_inv, pts_3dn)
        pts_3dn = np.matmul(Tr_inv, pts_3dn)
        pts_3dn = np.matmul(Tr, pts_3dn)
        xnn, ynn, znn = pts_3dn
        pts_3dn = [xnn, ynn, znn, 1]
        pts_3dn = np.matmul(R0_n, pts_3dn)
        pts_2dn = np.dot(P2_n, pts_3dn)
        pts_2dn = pts_2dn[:2] / pts_2dn[2:3]
        
        x2n, y2n= pts_2dn
        nnc_xy = [x2n, y2n]
        # Add amodel center
        
        
        ########################## nuScenes
        #camera_intrinsic = calib_tl
        #print(P2[0,0])
        #print(P2[0,2])
        print((bbox[0] + bbox[2]) / 2)
        alpha = _rot_y2alpha(rotation_y, (bbox[0] + bbox[2]) / 2, P2[0, 2], P2[0, 0] )
        ##########################
        

        ann = {'image_id': image_id,
               'id': int(len(ret['annotations']) + 1),
               'category_id': cat_id,
               'dim': dim,
               'bbox': _bbox_to_coco_bbox(bbox),
               'depth': location[2],
               'alpha': alpha,
               'truncated': truncated,
               'occluded': occluded,
               'location': location,
               'rotation_y': rotation_y,
               'amodel_center': nnc_xy,
               }
        ret['annotations'].append(ann)
        if DEBUG and tmp[0] != 'DontCare':
          box_3d = compute_box_3d(dim, location, rotation_y)
          box_2d = project_to_image(box_3d, calib)
          # print('box_2d', box_2d)
          image = draw_box_3d(image, box_2d)
          x = (bbox[0] + bbox[2]) / 2
          '''
          print('rot_y, alpha2rot_y, dlt', tmp[0], 
                rotation_y, alpha2rot_y(alpha, x, calib[0, 2], calib[0, 0]),
                np.cos(
                  rotation_y - alpha2rot_y(alpha, x, calib[0, 2], calib[0, 0])))
          '''
          depth = np.array([location[2]], dtype=np.float32)
          pt_2d = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                            dtype=np.float32)
          pt_3d = unproject_2d_to_3d(pt_2d, depth, calib)
          pt_3d[1] += dim[0] / 2
          #print('pt_3d', pt_3d)
          #print('location', location)
      if DEBUG:
        cv2.imshow('image', image)
        cv2.waitKey()


    print("# images: ", len(ret['images']))
    print("# annotations: ", len(ret['annotations']))
    # import pdb; pdb.set_trace()
    out_path = '{}/annotations/kitti_{}_{}.json'.format(DATA_PATH, SPLIT, split)
    json.dump(ret, open(out_path, 'w'))
  
