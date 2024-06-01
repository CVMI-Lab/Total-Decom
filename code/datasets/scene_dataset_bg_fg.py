import os
import torch
import torch.nn.functional as F
import numpy as np
import quaternion

import utils.general as utils
from utils import rend_util
from glob import glob
import cv2
import random
import json

# Dataset with monocular depth and normal
class SceneDatasetDN(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 center_crop_type='xxxx',
                 use_mask=False,
                 num_views=-1,
                 only_bg=True
                 ):

        self.instance_dir = os.path.join('../data', data_dir, 'scan{0}'.format(scan_id))

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        self.num_views = num_views
        assert num_views in [-1, 3, 6, 9]

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None
        
        def glob_data(data_dir):
            data_paths = []
            data_paths.extend(glob(data_dir))
            data_paths = sorted(data_paths)
            return data_paths
            
        image_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_rgb.png"))
        depth_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_depth.npy"))
        # depth_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_depth.png"))
        normal_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_normal.npy"))
        # ins_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_ins.png"))
        # feature_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*rgb_sam_CxHxW.pt"))

        # mask is only used in the replica dataset as some monocular depth predictions have very large error and we ignore it
        if use_mask:
            # The bg mask has four different values: 0, 1, 2, 3
            # 0: not avaliable, 1: floor, 2: wall, 3: ceiling, 4: layout 2
            mask_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*mask.npy"))
            if len(mask_paths) > len(image_paths):
                mask_split_paths = mask_paths[1::2]
                mask_error_paths = mask_paths[::2]
            else:
                mask_split_paths = mask_paths
                mask_error_paths = None

        else:
            mask_paths = None

        self.n_images = len(image_paths)
        
        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        scale_factors = scale_mats[0][0, 0]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            if scan_id == 3: 
            # Here we handcraftly add the rotation for the camera pose to train the scene of the replica room_2
                rotation_quat = np.array([0.992, -0.005, -0.075, 0.101])
                rotation_matrix = quaternion.as_rotation_matrix(quaternion.from_float_array(rotation_quat))
                rotation_matrix = np.concatenate([rotation_matrix, np.zeros((3, 1))], axis=1)
                rotation_matrix = np.concatenate([rotation_matrix, np.zeros((1, 4))], axis=0)
                print("Original Pose:", pose)
                pose = rotation_matrix @ pose
                print("Rotated Pose:", pose)
            elif scan_id == 2 and "tnt" in self.instance_dir: 
            # Here we handcraftly add the rotation for the camera pose to train the scene of the replica room_2
                rotation_quat = np.array([0.745, -0.667, 0.0, 0.0])
                rotation_matrix = quaternion.as_rotation_matrix(quaternion.from_float_array(rotation_quat))
                rotation_matrix = np.concatenate([rotation_matrix, np.zeros((3, 1))], axis=1)
                rotation_matrix = np.concatenate([rotation_matrix, np.zeros((1, 4))], axis=0)
                print("Original Pose:", pose)
                pose = rotation_matrix @ pose
                print("Rotated Pose:", pose)

            # because we do resize and center crop 384x384 when using omnidata model, we need to adjust the camera intrinsic accordingly
            if center_crop_type == 'center_crop_for_replica':
                scale = 384 / 680
                offset = (1200 - 680 ) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'center_crop_for_tnt':
                scale = 384 / 540
                offset = (960 - 540) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'center_crop_for_dtu':
                scale = 384 / 1200
                offset = (1600 - 1200) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'padded_for_dtu':
                scale = 384 / 1200
                offset = 0
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'no_crop':  # for scannet dataset, we already adjust the camera intrinsic duing preprocessing so nothing to be done here
                pass
            else:
                raise NotImplementedError
            
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
            
        self.depth_images = []
        self.normal_images = []
        self.feature_images = []

        # for dpath, npath, fpath in zip(depth_paths, normal_paths, feature_paths):
        # for dpath, npath, spath in zip(depth_paths, normal_paths, sem_paths):
        # for dpath, npath, fpath in zip(depth_paths, normal_paths, feature_paths):
        for dpath, npath in zip(depth_paths, normal_paths):
            depth = np.load(dpath)
            # depth = cv2.imread(dpath, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
            # NOTE: I have checked here and the loaded depth scale is right
            # depth = depth / scale_factors
            self.depth_images.append(torch.from_numpy(depth.reshape(-1, 1)).float())
        
            normal = np.load(npath)
            normal = normal.reshape(3, -1).transpose(1, 0)
            # important as the output of omnidata is normalized
            normal = normal * 2. - 1.
            self.normal_images.append(torch.from_numpy(normal).float())

            # Load the semantic
            # sem = cv2.imread(spath, cv2.IMREAD_GRAYSCALE)

            # we upsample the feature to 384x384
            # feature = torch.load(fpath)
            # # We should assure that the feature is a 4D tensor
            # if len(feature.size()) == 3:
            #     feature = feature.unsqueeze(0)

            # feature = torch.nn.functional.interpolate(feature.float(), size=(384, 384), mode='nearest')
                                                  
            # feature = feature.reshape(256, -1).transpose(1, 0).cpu()
            # self.feature_images.append(feature)

        # NOTE: We need to know whether we need to reconstruct the scene in 3 layout
        # load mask
        # self.layout_1 = []
        # self.layout_2 = []
        self.sem_images = []
        self.layout_images = []
        for m_path in mask_split_paths:
        # for m_path, i_path in zip(mask_split_paths, ins_paths):
            mask = np.load(m_path)
            sem = mask.copy()
            # Here, the sem images represent the foreground and the background
            # The label 1 represent the floor and the label 2 represent the wall
            # The semantic label only for wall, floor, ceiling
            sem[sem == 0] = 4
            sem -= 1
            self.sem_images.append(torch.from_numpy(sem.reshape(-1, 1)).float())
            # ins_label = cv2.imread(i_path, cv2.IMREAD_GRAYSCALE)
            mask[mask == 4] = 0
            mask[mask != 0] = 1
            self.layout_images.append(torch.from_numpy(mask.reshape(-1, 1)).float())

        self.gt_masks = []
        if mask_error_paths is not None:
            for m_path in mask_error_paths:
                mask = np.load(m_path)
                self.gt_masks.append(torch.from_numpy(mask.reshape(-1, 1)).float())
        else:
            for depth in self.depth_images:
                mask = torch.ones_like(depth)
                self.gt_masks.append(mask)


    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if self.num_views >= 0:
            image_ids = [25, 22, 28, 40, 44, 48, 0, 8, 13][:self.num_views]
            idx = image_ids[random.randint(0, self.num_views - 1)]
        
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx],
        }
        
        ground_truth = {
            "rgb": self.rgb_images[idx],
            "depth": self.depth_images[idx],
            "normal": self.normal_images[idx],
            "layout": self.layout_images[idx],
            "sem": self.sem_images[idx],
            # "feature": self.feature_images[idx],
            "mask": self.gt_masks[idx],
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            ground_truth["full_rgb"] = self.rgb_images[idx]
            ground_truth["normal"] = self.normal_images[idx][self.sampling_idx, :]
            ground_truth["depth"] = self.depth_images[idx][self.sampling_idx, :]
            ground_truth["full_depth"] = self.depth_images[idx]
            ground_truth["layout"] = self.layout_images[idx][self.sampling_idx, :]
            ground_truth["mask"] = self.gt_masks[idx][self.sampling_idx, :]
            sample["uv"] = uv[self.sampling_idx, :]
            ground_truth["sem"] = self.sem_images[idx][self.sampling_idx, :]
            # ground_truth["feature"] = self.feature_images[idx][self.sampling_idx, :]

        return idx, sample, ground_truth

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

    def change_sampling_idx(self, sampling_size, idx=None, patch_sampling=False):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            if patch_sampling:
                patch_size = 32
                n_patches = 1

                # Randomly select the masked points
                patch_mask_bound = np.zeros((384, 384))
                patch_mask_bound[:(self.img_res[0] - patch_size + 1), :(self.img_res[1] - patch_size + 1)] = 1
                patch_mask = np.logical_and((self.layout_2[idx] == 0).reshape(384, 384), patch_mask_bound)
                candidate_uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
                candidate_uv = candidate_uv[:, patch_mask.bool().numpy()]
                if candidate_uv.shape[1] > 0:
                    selected_ind = np.random.randint(0, candidate_uv.shape[1])
                    xy0 = candidate_uv[:, selected_ind].reshape(n_patches, 1, -1)
                else:
                    x0 = np.random.randint(0, self.img_res[1] - patch_size + 1, size=(n_patches, 1, 1))
                    y0 = np.random.randint(0, self.img_res[0] - patch_size + 1, size=(n_patches, 1, 1))
                    xy0 = np.concatenate([x0, y0], axis=-1)
                patch_idx = xy0 + np.stack(np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing='xy'),axis=-1).reshape(1, -1, 2)
                uv0 = torch.from_numpy(patch_idx).float().reshape(-1, 2).long()

                patch_mask_matrix = torch.zeros(self.img_res[0], self.img_res[1])
                patch_mask_matrix[uv0[:,1], uv0[:,0]] = 1       # the index here important
                patch_mask_matrix = patch_mask_matrix.bool()

                patch_mask_lin = patch_mask_matrix.reshape(-1)      # 384*384 long tensor with 1024 true
                tmp = torch.linspace(0,384*384,384*384).long()
                tmp = tmp[patch_mask_lin]

                self.sampling_idx = tmp  # the order here is important, the sampled pixel will be iterate over u(row), and then v(column)
            else:
                self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']


    def compute_scale_and_shift_np(self, prediction, target, mask):
        # system matrix: A = [[a_00, a_01], [a_10, a_11]]
        a_00 = np.sum(mask * prediction * prediction, (1, 2))
        a_01 = np.sum(mask * prediction, (1, 2))
        a_11 = np.sum(mask, (1, 2))

        # right hand side: b = [b_0, b_1]
        b_0 = np.sum(mask * prediction * target, (1, 2))
        b_1 = np.sum(mask * target, (1, 2))

        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        x_0 = np.zeros_like(b_0)
        x_1 = np.zeros_like(b_1)

        det = a_00 * a_11 - a_01 * a_01
        valid = det.nonzero()

        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        return x_0, x_1
