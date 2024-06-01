# Import related libs and claim the dependencies

# GPU control
import os
import argparse

# Additional system path
import sys
sys.path.append("..")

# Basic operations
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
import cv2
from scipy import ndimage
import copy
import torchvision

# 3D geometry related operations
import open3d as o3d

# 2D image related operations
from PIL import Image
from PIL import ImageOps
from skimage.transform import resize

# Visualization related operations
import matplotlib.pyplot as plot

# Generate point cloud information for easy visualization
from plyfile import PlyData, PlyElement

# NeRF-related operation
from pyhocon import ConfigFactory
import utils.general as utils
import utils.plots as plt
from utils import rend_util
from utils.plots import get_all_vert_feature

# SAM-related operation
# from segment_anything import sam_model_registry, SamPredictor

# Statistical operations
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree

# Evaluation-realted operations
from scipy.spatial import cKDTree
import trimesh

# Visualization related functions

# Plot a heatmap to visualize the feature
def plot_heatmap(data, title):
    plot.imshow(data, cmap='viridis', interpolation='nearest')
    plot.colorbar()
    plot.title(title)
    plot.axis('off')
    plot.show()

# Save the selected point set in a .ply format

def save_pc(data, filename):
    points = data
    vertex = np.zeros(points.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex['x'] = points[:, 0]
    vertex['y'] = points[:, 1]
    vertex['z'] = points[:, 2]
    vertex_element = PlyElement.describe(vertex, 'vertex')
    plydata = PlyData([vertex_element])
    plydata.write(filename)
    
# Evaluation function inherited from MonoSDF, for quick output retrieving

def evaluate(**kwargs):
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    eval_rendering = kwargs['eval_rendering']

    scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else conf.get_int('dataset.scan_id', default=-1)

    dataset_conf = conf.get_config('dataset')
    if kwargs['scan_id'] != -1:
        dataset_conf['scan_id'] = kwargs['scan_id']
    
    # use all images for evaluation
    dataset_conf['num_views'] = -1

    eval_dataset = utils.get_class(conf.get_string('train.dataset_class'))(**dataset_conf)

    conf_model = conf.get_config('model')
    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf_model)
    if torch.cuda.is_available():
        model.cuda()

    # settings for camera optimization
    scale_mat = eval_dataset.get_scale_mat()
    if True:
        if eval_rendering:
            eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      collate_fn=eval_dataset.collate_fn
                                                      )
            total_pixels = eval_dataset.total_pixels
            img_res = eval_dataset.img_res
            split_n_pixels = conf.get_int('train.split_n_pixels', 10000)

            saved_model_state = torch.load(str(kwargs['checkpoint']))
    
            # deal with multi-gpu training model
            if list(saved_model_state["model_state_dict"].keys())[0].startswith("module."):
                saved_model_state["model_state_dict"] = {k[7:]: v for k, v in saved_model_state["model_state_dict"].items()}

            model.load_state_dict(saved_model_state["model_state_dict"], strict=True)
            print("evaluating...")

            model.eval()
            
            
            if True:
                intrinsics_list = []
                pose_list = []
                i = 0
                for data_index, (indices, model_input, ground_truth) in enumerate(eval_dataloader):
                    if i < 80 or i > 85:
                        i += 1
                        continue
                    model_input["intrinsics"] = model_input["intrinsics"].cuda()
                    model_input["uv"] = model_input["uv"].cuda()
                    model_input['pose'] = model_input['pose'].cuda()

                    split = utils.split_input(model_input, total_pixels, n_pixels=split_n_pixels)
                    res = []
                    dres = []
                    lres = []
                    sam_res = []
                    for s in tqdm(split):
                        torch.cuda.empty_cache()
                        out = model(s, indices)
                        res.append({
                            'rgb_values': out['rgb_values'].detach(),
                        })
                        dres.append({
                            'depth_values': out['depth_values'].detach(),
                        })
                        lres.append({
                            'layout_probs': out['layout_probs'].detach(),
                        })
                        sam_res.append({
                            'feature_values': out['feature_values'].detach(),
                        })
                    batch_size = ground_truth['rgb'].shape[0]
                    model_outputs = utils.merge_output(res, total_pixels, batch_size)
                    model_embedding = utils.merge_output(sam_res, total_pixels, batch_size)
                    model_depth = utils.merge_output(dres, total_pixels, batch_size)
                    model_layout = utils.merge_output(lres, total_pixels, batch_size)
                    image_embedding = model_embedding['feature_values']
                    # Modified: 
                    image_embedding = image_embedding.reshape(384, 384, 256)
                    image_embedding = image_embedding.transpose(1,2).transpose(0,1).unsqueeze(0)
                    # resized_feat = resize(image_embedding.cpu().numpy(), (1, 256, 64, 64), anti_aliasing=True, mode='reflect')

                    # image_embedding_align = torch.from_numpy(resized_feat).cuda()
                    # vis = np.array(image_embedding_align.cpu().reshape(256,64,64))
                    # vis_heat = np.sum(vis, axis=0)
                    # plot_heatmap(vis_heat, 'Heatmap of vis')

                    images_dir = kwargs['output_dir']
                    os.makedirs(images_dir, exist_ok=True)
                    torch.save(image_embedding, '{0}/eval_{1}.pt'.format(images_dir,'%03d' % indices[0]))

                    rgb_eval = model_outputs['rgb_values']
                    rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)
                    rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
                    rgb_eval = rgb_eval.transpose(1, 2, 0)
                    img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
                    img.save('{0}/eval_{1}.png'.format(images_dir,'%03d' % indices[0])) 
                    
                    depth_eval = model_depth['depth_values']
                    depth_eval = depth_eval.reshape(batch_size, total_pixels, 1)
                    depth_eval = plt.lin2img(depth_eval, img_res).detach().cpu().numpy()[0]
                    depth_eval = depth_eval.transpose(1, 2, 0)
                    # plot.imshow(depth_eval, cmap='gray')
                    # plot.show()
                    np.save('{0}/eval_depth_{1}.npy'.format(images_dir,'%03d' % indices[0]), depth_eval)
                    
                    layout_eval = model_layout['layout_probs']
                    layout_pred = torch.argmax(model_layout['layout_probs'], -1).reshape(batch_size, 147456)
                    sem_maps_plot = plt.lin2img(layout_pred[..., None], img_res)
                    sem_maps_plot = sem_maps_plot.expand(-1, 3, -1, -1)
                    tensor = torchvision.utils.make_grid(sem_maps_plot,
                                                        scale_each=False,
                                                        normalize=False,
                                                        nrow=1).cpu().detach().numpy()
                    tensor = tensor.transpose(1, 2, 0)
                    layout_eval = tensor[:, :, 0]
                    np.save('{0}/eval_layout_{1}.npy'.format(images_dir,'%03d' % indices[0]), layout_eval)

                    # Record for parameters
                    intrinsics = model_input["intrinsics"]
                    pose = model_input["pose"]
                    intrinsics = intrinsics.squeeze().cpu().numpy()[:3, :3]
                    pose = pose.squeeze().cpu().numpy()
                    intrinsics_list.append(intrinsics)
                    pose_list.append(pose)
                    rendering_image = rgb_eval
                    feature = image_embedding
                    depth = depth_eval
                    intrinsics = np.array(intrinsics_list)
                    pose = np.array(pose_list)
                    i += 1
                return intrinsics, pose, rendering_image, feature, depth, layout_eval

# SAM front-end and Mask Selection
# SAM Helper functions
def show_mask(mask, ax, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=25):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)    
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export ScanNet rendering features for decomposition')
    parser.add_argument('--conf', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--scan_id', type=int, default=1)
    args = parser.parse_args()

    intrinsics, pose, _, _, _, _ = evaluate(
        conf=args.conf,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        scan_id=args.scan_id,
        resolution=512,
        world_space=False,
        eval_rendering=True,
    )
    np.save(os.path.join(args.output_dir, 'intrinsics.npy'), intrinsics)
    np.save(os.path.join(args.output_dir, 'pose.npy'), pose)
