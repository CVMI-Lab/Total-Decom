import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from PIL import Image
from glob import glob
import random
from src.utils.load_utils import glob_imgs, load_K_Rt_from_P
from src.utils.cam_utils import Camera
from src.utils.graphics_utils import *

class SceneDataset(torch.utils.data.Dataset):
    # Convert the cameras.npz file to the gaussian splatting format

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 ):

        self.instance_dir = os.path.join(os.path.expanduser(data_dir), 'scan{0}'.format(scan_id))

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        def glob_data(data_dir):
            data_paths = []
            data_paths.extend(glob(data_dir))
            data_paths = sorted(data_paths)
            return data_paths
            
        image_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_rgb.png"))
        self.n_images = len(image_paths)

        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.cameras = []
        self.poses = []
        idx = 0
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.poses.append(pose)

            # NOTE: For the monosdf dataset, we need to reshape the intrinsics
            scale = 384 / 680
            offset = (1200 - 680) * 0.5
            intrinsics[0, 2] -= offset
            intrinsics[:2, :] *= scale

            # NOTE: We do not need any operations here for the gaussian splatting codebase
            # For the gaussian splatting original papers, 
            # we need to convert the c2w coordinate to the w2c coordinate
            w2c = np.linalg.inv(pose)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            image_path = image_paths[idx]
            image_name = Path(image_path).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4]
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            focal_x = intrinsics[0, 0]
            focal_y = intrinsics[1, 1]

            FovX = focal2fov(focal_x, image.size[0])
            FovY = focal2fov(focal_y, image.size[1])
            image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).permute(2, 0, 1)

            self.cameras.append(Camera(colmap_id=idx, R=R, T=T, FoVx=FovX, FoVy=FovY, 
                                       image=image, gt_alpha_mask=None, image_name=image_name, 
                                       uid=id, data_device='cuda'))

            idx += 1
            
    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):

        return self.cameras[idx]

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)
