from nuscenes.nuscenes import NuScenes


nusc = NuScenes(version="v1.0-trainval", dataroot="../../data/nuscenes/v1.0-trainval", verbose=True)
nusc.list_scenes()