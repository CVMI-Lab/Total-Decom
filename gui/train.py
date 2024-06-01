import os
from pathlib import Path
import cv2
import time
import tqdm
import numpy as np
import trimesh
import dearpygui.dearpygui as dpg
import nvdiffrast.torch as dr

import torch
import torch.nn.functional as F

from segment_anything import sam_model_registry, SamPredictor
from kiui.mesh import Mesh
from kiui.op import safe_normalize

from src.dataset.replica import SceneDataset
from src.rendering.gs_renderer import Renderer, MiniCam

from src.utils.cam_utils import orbit_camera, OrbitCamera
from src.utils.loss_utils import l1_loss, ssim
from src.utils.graphics_utils import MeshRenderer
# from grid_put import mipmap_linear_grid_put_2d
# from mesh import Mesh, safe_normalize

# Dependencies or Operations related to segmentation
# Warning: Corresponding operations may differ based on different lib

# Statistical operations
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
from scipy.ndimage import binary_dilation, binary_erosion
# Visualization related operations
import matplotlib.pyplot as plot
# 3D geometry related operations
import open3d as o3d
from plyfile import PlyData, PlyElement
# Basic operations
import pandas as pd
from scipy import ndimage
import copy
import torchvision
from collections import defaultdict
from pyhocon import ConfigFactory
# import src.utils.general as utils
from collections import Counter


def require_file(path, option_name):
    path = Path(os.path.expanduser(str(path)))
    if not path.is_file():
        raise FileNotFoundError(
            f"{option_name} does not point to a file: {path}. "
            "Update the selected YAML config before launching the GUI."
        )
    return str(path)


def require_directory(path, option_name):
    path = Path(os.path.expanduser(str(path)))
    if not path.is_dir():
        raise FileNotFoundError(
            f"{option_name} does not point to a directory: {path}. "
            "Update the selected YAML config before launching the GUI."
        )
    return str(path)

# A quick debug function
def save_pc(data, filename):
    points = data
    vertex = np.zeros(points.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex['x'] = points[:, 0]
    vertex['y'] = points[:, 1]
    vertex['z'] = points[:, 2]
    vertex_element = PlyElement.describe(vertex, 'vertex')
    plydata = PlyData([vertex_element])
    plydata.write(filename)

# Core function: Surface Growing
def find_adjacent_vertices(vertex_index, adj_list, triangles):
    adjacent_vertices = set(adj_list[vertex_index])
    return np.array(list(adjacent_vertices))
# Region growing function
def surface_growing(coord, feature, indices, dilated_indices, adj_list, vert, surface_graph, selected, max_iterations=10000, threshold=1.0, accuracy=0.01):
    """
    Perform surface growing based on given coordinates and features using seed points.

    Args:
        coord (numpy.ndarray): Array of size [n, 3] containing the coordinates of the points.
        feature (torch.Tensor): Tensor of size [n, 256] containing the feature vectors for each point.
        indices (list): List of indices of seed points to start the region growing from.
        max_iterations (int): Maximum number of iterations to perform. Default is 1000.
        threshold (float): Feature similarity threshold to consider two points as similar. Default is 0.9.

    Returns:
        numpy.ndarray: Array of size [n,] containing the labels of the segmented points.
    """
    t1 = time.time()
    labels = np.copy(selected)
    count = np.count_nonzero(labels[dilated_indices.tolist()])
    current_label = 0
    potential = []
    for seed_idx in indices:
        if labels[seed_idx] != 0 and labels[seed_idx] != -1:
            continue
        current_label = 1
        labels[seed_idx] = current_label
        queue = [seed_idx]
        for iteration in range(max_iterations):
            if iteration % 50 == 0:
                count = np.count_nonzero(labels[dilated_indices.tolist()])
                if count/len(labels[dilated_indices.tolist()]) > 0.1: #or (count > 30 and threshold < 0.999):
                    t2 =time.time()
                    print(threshold, t2-t1, len(vert[labels!=0]), len(potential), count/len(labels[dilated_indices.tolist()]))
                    return np.array([])
            if not queue:
                break
            idx = queue.pop(0)
            seed_coord = coord[idx]
            seed_feature = feature[idx]

            # Find neighbors
            neighbors = find_adjacent_vertices(idx, adj_list, surface_graph)

            # Check if neighbors are similar in feature space and assign them the same label if they are.
            for neighbor_idx in neighbors:
                if labels[neighbor_idx] == 0:
                    neighbor_feature = feature[neighbor_idx]
                    similarity = np.dot(seed_feature, neighbor_feature) / (np.linalg.norm(seed_feature) * np.linalg.norm(neighbor_feature))
                    if similarity > threshold:
                        labels[neighbor_idx] = current_label
                        queue.append(neighbor_idx)
                        adj_list[idx].remove(neighbor_idx)
                        adj_list[neighbor_idx].remove(idx)
                    else:
                        labels[idx] = -1
            if labels[idx] == -1:
                potential.append(idx)
    # Step_in check
    t2 =time.time()
    count = np.count_nonzero(labels[dilated_indices.tolist()])
    print(threshold, t2-t1, len(vert[labels!=0]), len(potential), count/len(labels[dilated_indices.tolist()]))
    # To deal with uncontrollable growing
    if len(vert[labels!=0]) - len(vert[selected!=0]) > 10000 and threshold < 1 - accuracy * 3:
        return selected
    if len(potential) == 0:
        return labels
    step_in_label = surface_growing(vert, feature, potential, dilated_indices, adj_list, vert, surface_graph, labels, threshold=threshold-accuracy, accuracy=accuracy)
    if step_in_label.shape[0] == 0:
        return labels
    else:
        return step_in_label

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.gaussain_scale_factor = 1

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop

        self.epochs = opt.epochs
        # NOTE: Generate the dataset here
        data_dir = require_directory(self.opt.data_dir, "data_dir")
        load_path = require_file(self.opt.load, "load")
        foreground_mesh = require_file(self.opt.foreground_mesh, "foreground_mesh")
        vertex_features = require_file(self.opt.vertex_features, "vertex_features")
        sam_checkpoint = require_file(self.opt.sam_checkpoint, "sam_checkpoint")
        os.makedirs("./tmp/raw_decomposition", exist_ok=True)

        self.training_data = SceneDataset(data_dir, [self.opt.img_H, self.opt.img_W], self.opt.scan_id)
        self.loaded_mesh = Mesh.load_trimesh(load_path)
        self.fg_mesh = Mesh.load_trimesh(foreground_mesh)
        self.glctx = dr.RasterizeCudaContext()
        self.light_dir = np.array([0, 0])
        self.ambient_ratio = 0.7
        self.bg_color = torch.ones(3, dtype=torch.float32, device=self.device)
        # loaded_mesh = trimesh.load_mesh(self.opt.load)
        # self.mesh_renderer = MeshRenderer(mesh=loaded_mesh)

        # User-friendly parameters
        self.decompose = False
        self.finish = False
        self.extracted = None
        self.surface = True
        self.selected_triangles = None
        self.grow = False
        self.undo = False
        self.operation = True
        self.change = 0
        self.history = []
        self.prompt_point = []
        self.prompt_label = []
        self.clusters = []
        # Paramters for scene decomposition
        # WARNING: Precomputation may takes time
        print("loading mesh data...")
        print()
        # Clustering for obtaining the connected components
        processed_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(self.fg_mesh.v.cpu().numpy().copy()),
            triangles=o3d.utility.Vector3iVector(self.fg_mesh.f.cpu().numpy().copy()),
        )
        processed_vert = np.asarray(processed_mesh.vertices)
        processed_triangles = np.asarray(processed_mesh.triangles)
        processed_min_vert, processed_max_vert = processed_vert.min(axis=0), processed_vert.max(axis=0)
        processed_mesh.compute_vertex_normals()
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (
                processed_mesh.cluster_connected_triangles())
        processed_triangle_clusters = np.asarray(triangle_clusters)
        processed_cluster_n_triangles = np.asarray(cluster_n_triangles)
        processed_cluster_area = np.asarray(cluster_area)
        for idx in range(processed_triangle_clusters.max()+1):
            mesh_component = copy.deepcopy(processed_mesh)
            triangles_to_remove = processed_triangle_clusters != idx
            mesh_component.remove_triangles_by_mask(triangles_to_remove)
            o3d.io.write_triangle_mesh("./tmp/raw_decomposition/"+str(idx)+".ply", mesh_component)
        self.vert = processed_vert
        self.triangles = processed_triangles
        self.triangle_clusters = processed_triangle_clusters
        self.cluster_num = processed_triangle_clusters.max()+1
        self.fg_mesh_o3d = processed_mesh
        self.mask = np.zeros((self.H, self.W)).astype(bool)
        # Construct an adj list
        self.adj_list = defaultdict(list)
        for triangle in self.triangles:
            for i in range(3):
                self.adj_list[triangle[i]].append(triangle[(i+1)%3])
                self.adj_list[triangle[i]].append(triangle[(i+2)%3])
        
        print("Load the reconstruction parameters for enabling interaction...")
        print()
        self.all_vert_feature = torch.load(vertex_features, map_location="cpu")
        if len(self.all_vert_feature) != len(self.vert):
            raise ValueError(
                "vertex_features must contain one feature vector per foreground-mesh vertex "
                f"({len(self.vert)} vertices, got {len(self.all_vert_feature)} features)."
            )
        

        # NOTE: Load the SAM Model
        model_type = self.opt.sam_model_type
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        self.mouse_loc = None
        self.left_mouse_pos = None
        self.right_mouse_pos = None
        
        # NOTE: Pay attention here
        # override if provide a checkpoint
        if self.opt.load is not None:
            self.renderer.initialize(load_path, load_ply=self.opt.load_ply)
        else:
            # initialize gaussians to a blob
            self.renderer.initialize(num_pts=self.opt.num_pts)

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    @torch.no_grad()
    def get_all_vert_feature(points, feature):
        selected_points = torch.from_numpy(points.astype(np.float32)).to(self.device)
        f = []
        for _, pnts in enumerate(torch.split(selected_points, 100000, dim=0)):
                f.append(feature(pnts))
        selected_feature = torch.cat(f, axis=0)
        return selected_feature
    
    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def prepare_train(self):

        self.step = 0

        # setup training
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        # self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer


        self.random_choice = list(np.random.choice(len(self.training_data), 
                                                   len(self.training_data), 
                                                   replace=False))


    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        if len(self.random_choice) == 0:
            self.random_choice = list(np.random.choice(len(self.training_data), 
                                                       len(self.training_data), 
                                                       replace=False))
        idx = self.random_choice.pop()
        cur_cam = self.training_data[idx]
        self.step += 1

        if self.step % 1000 == 0:
            self.renderer.gaussians.oneupSHdegree()
        # step_ratio = min(1, self.step / self.opt.iters)

        # update lr
        self.renderer.gaussians.update_learning_rate(self.step)
        out = self.renderer.render(cur_cam)

        # rgb loss
        image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
        gt_image = cur_cam.original_image.unsqueeze(0).to(self.device) # [1, 3, H, W] in [0, 1]
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # optimize step
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # NOTE: We may not need the operation here
        # NOTE: We must know that the quality will be decreased without the densify and prune operation
        # densify and prune
        if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
            viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
            self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if self.step % self.opt.densification_interval == 0:
                size_threshold = 20 if self.step > self.opt.opacity_reset_interval else None
                self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.005, extent=1.1, max_screen_size=size_threshold)
            
        # if self.step % self.opt.opacity_reset_interval == 0 and self.step < 30000:
        #     self.renderer.gaussians.reset_opacity()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

        if self.gui:
            dpg.set_value("_log_train_time", f"{t:.4f}ms")
            dpg.set_value(
                "_log_train_log",
                f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}",
            )

        # dynamic train steps (no need for now)
        # max allowed train time per-frame is 500 ms
        # full_t = t / self.train_steps * 16
        # train_steps = min(16, max(4, int(16 * 500 / full_t)))
        # if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
        #     self.train_steps = train_steps

    @torch.no_grad()
    def test_step(self):
        # ignore if no need to update
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            # render image

            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )

            out = self.renderer.render(cur_cam, self.gaussain_scale_factor)
            if self.mode == "mesh":
                if self.selected_triangles is not None:
                    pose = torch.from_numpy(self.cam.pose_mesh.astype(np.float32)).cuda()
                    proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).cuda()
                    v_cam = torch.matmul(F.pad(self.fg_mesh.v, pad=(0, 1), mode='constant', value=1.0), 
                                        torch.inverse(pose).T).float().unsqueeze(0)
                    v_clip = v_cam @ proj.T
                    v_clip[..., :1] -= 0.5 / self.H
                    v_clip[..., 1:2] += 0.5 / self.H

                    rast, rast_db = dr.rasterize(self.glctx, v_clip, self.fg_mesh.f, (self.H, self.W))
                    faces = rast[..., 3:].cpu().numpy().reshape(-1).astype(int) - 1
                    
                    alpha = (rast[..., 3:] > 0).float() # Check this
                    alpha = dr.antialias(alpha, rast, v_clip, self.fg_mesh.f).squeeze(0).clamp(0, 1) # [H, W, 3]

                    # albedo, _ = dr.interpolate(self.loaded_mesh.albedo.unsqueeze(0).contiguous(), rast, self.loaded_mesh.f)
                    albedo, _ = dr.interpolate(torch.ones_like(v_clip), rast, self.fg_mesh.f)

                    albedo = torch.where(rast[..., 3:] > 0, albedo, torch.tensor(0).to(albedo.device)) # remove background
                    albedo = dr.antialias(albedo, rast, v_clip, self.fg_mesh.f).clamp(0, 1) # [1, H, W, 3]
                    normal, _ = dr.interpolate(self.fg_mesh.vn.unsqueeze(0).contiguous(), rast, self.fg_mesh.fn)
                    normal = safe_normalize(normal)

                    light_d = np.deg2rad(self.light_dir)
                    light_d = np.array([
                        np.sin(light_d[0]) * np.sin(light_d[1]),
                        np.cos(light_d[0]),
                        np.sin(light_d[0]) * np.cos(light_d[1]),
                    ], dtype=np.float32)
                    light_d = torch.from_numpy(light_d).to(albedo.device)
                    lambertian = self.ambient_ratio + (1 - self.ambient_ratio)  * (normal @ light_d).float().clamp(min=0)
                    lambertian_inverse = self.ambient_ratio + (1 - self.ambient_ratio)  * (normal @ -light_d).float().clamp(min=0)

                    # albedo = (albedo * lambertian.unsqueeze(-1)) * alpha + self.bg_color * (1 - alpha)
                    albedo = (albedo * lambertian.unsqueeze(-1) * lambertian_inverse.unsqueeze(-1))[0, :,:,:3] * alpha + self.bg_color * (1 - alpha)
                    # buffer = albedo[0].detach().cpu().numpy()
                    albedo = albedo.cpu().numpy()
                    filter = np.in1d(faces.reshape(self.H, self.W), self.selected_triangles).reshape(self.H, self.W)
                    albedo[~filter] = np.array([1,1,1])
                        
                    self.buffer_image = albedo
                else:
                    pose = torch.from_numpy(self.cam.pose_mesh.astype(np.float32)).cuda()
                    proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).cuda()
                    v_cam = torch.matmul(F.pad(self.loaded_mesh.v, pad=(0, 1), mode='constant', value=1.0), 
                                        torch.inverse(pose).T).float().unsqueeze(0)
                    v_clip = v_cam @ proj.T
                    v_clip[..., :1] -= 0.5 / self.H
                    v_clip[..., 1:2] += 0.5 / self.H

                    rast, rast_db = dr.rasterize(self.glctx, v_clip, self.loaded_mesh.f, (self.H, self.W))

                    alpha = (rast[..., 3:] > 0).float() # Check this
                    alpha = dr.antialias(alpha, rast, v_clip, self.loaded_mesh.f).squeeze(0).clamp(0, 1) # [H, W, 3]

                    # albedo, _ = dr.interpolate(self.loaded_mesh.albedo.unsqueeze(0).contiguous(), rast, self.loaded_mesh.f)
                    albedo, _ = dr.interpolate(torch.ones_like(v_clip), rast, self.loaded_mesh.f)

                    albedo = torch.where(rast[..., 3:] > 0, albedo, torch.tensor(0).to(albedo.device)) # remove background
                    albedo = dr.antialias(albedo, rast, v_clip, self.loaded_mesh.f).clamp(0, 1) # [1, H, W, 3]
                    normal, _ = dr.interpolate(self.loaded_mesh.vn.unsqueeze(0).contiguous(), rast, self.loaded_mesh.fn)
                    normal = safe_normalize(normal)

                    light_d = np.deg2rad(self.light_dir)
                    light_d = np.array([
                        np.sin(light_d[0]) * np.sin(light_d[1]),
                        np.cos(light_d[0]),
                        np.sin(light_d[0]) * np.cos(light_d[1]),
                    ], dtype=np.float32)
                    light_d = torch.from_numpy(light_d).to(albedo.device)
                    lambertian = self.ambient_ratio + (1 - self.ambient_ratio)  * (normal @ light_d).float().clamp(min=0)
                    lambertian_inverse = self.ambient_ratio + (1 - self.ambient_ratio)  * (normal @ -light_d).float().clamp(min=0)

                    # albedo = (albedo * lambertian.unsqueeze(-1)) * alpha + self.bg_color * (1 - alpha)
                    albedo = (albedo * lambertian.unsqueeze(-1) * lambertian_inverse.unsqueeze(-1))[0, :,:,:3] * alpha + self.bg_color * (1 - alpha)
                    # buffer = albedo[0].detach().cpu().numpy()
                    self.buffer_image = albedo.cpu().numpy()
                
            elif self.mode == "layout":
                pose = torch.from_numpy(self.cam.pose_mesh.astype(np.float32)).cuda()
                proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).cuda()
                v_cam = torch.matmul(F.pad(self.fg_mesh.v, pad=(0, 1), mode='constant', value=1.0), 
                                     torch.inverse(pose).T).float().unsqueeze(0)
                v_clip = v_cam @ proj.T
                v_clip[..., :1] -= 0.5 / self.H
                v_clip[..., 1:2] += 0.5 / self.H

                rast, rast_db = dr.rasterize(self.glctx, v_clip, self.fg_mesh.f, (self.H, self.W))
                alpha = (rast[..., 3:] > 0).float()
                buffer_image = alpha.reshape(self.H, self.W, 1).transpose(0, 2).transpose(1,2)
                buffer_image = buffer_image.repeat(3, 1, 1)
                self.buffer_image = (
                    buffer_image.permute(1, 2, 0)
                    .contiguous()
                    .clamp(0, 1)
                    .contiguous()
                    .detach()
                    .cpu()
                    .numpy()
                )
            
            elif self.mode == "decompose":
                buffer_image = out["image"]  # [3, H, W]
                if self.operation:
                    sam_image = buffer_image.permute(1, 2, 0).detach().cpu().numpy() * 255
                    self.predictor.set_image(sam_image.astype(np.uint8))

                    pose = torch.from_numpy(self.cam.pose_mesh.astype(np.float32)).cuda()
                    proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).cuda()
                    v_cam = torch.matmul(F.pad(self.fg_mesh.v, pad=(0, 1), mode='constant', value=1.0), 
                                        torch.inverse(pose).T).float().unsqueeze(0)
                    v_clip = v_cam @ proj.T
                    v_clip[..., :1] -= 0.5 / self.H
                    v_clip[..., 1:2] += 0.5 / self.H

                    rast, rast_db = dr.rasterize(self.glctx, v_clip, self.fg_mesh.f, (self.H, self.W))
                    alpha = (rast[..., 3:] > 0).float()
                    faces = rast[..., 3:].cpu().numpy().reshape(-1).astype(int) - 1
                    # faces[np.where(faces == -1)[0]] = len(self.triangle_clusters) + 10
                    mask = np.zeros((self.H, self.W)).astype(bool)
                    flip_mask = np.zeros((self.H, self.W)).astype(bool)
                    sub_mask = np.zeros((self.H, self.W)).astype(bool)
                    ccid = self.triangle_clusters[faces]
                    ccid[np.where(faces == -1)[0]] = self.cluster_num + 10
                    ccid = ccid.reshape(self.H, self.W)
                    layout = alpha.reshape(self.H, self.W, 1).cpu().numpy()
                    if self.left_mouse_pos is not None and self.left_mouse_pos[0] < self.W and self.left_mouse_pos[1] < self.H and self.decompose:
                        if self.surface:
                            self.selected_triangles = None
                            self.change = 1
                            self.prompt_point.append([self.left_mouse_pos[0], self.left_mouse_pos[1]])
                            self.prompt_label.append(1)
                            input_point = np.array(self.prompt_point)
                            input_label = np.array(self.prompt_label)
                            masks, scores, logits = self.predictor.predict(
                                                        point_coords=input_point,
                                                        point_labels=input_label,
                                                        multimask_output=False,)
                            index = np.argmax(scores)
                            mask = masks[index]
                            mask = np.logical_or(self.mask, mask)
                        else:
                            self.change = 2
                            input_point = np.array([[self.left_mouse_pos[0], self.left_mouse_pos[1]]])
                            input_label = np.array([1])
                            input_point = input_point.astype(int)
                            pid = ccid[input_point[0][1]][input_point[0][0]]
                            mask = ccid == pid
                            self.clusters.append(pid)
                            self.mask = np.logical_or(self.mask, mask)
                            mask = np.logical_or(self.mask, mask)
                            if self.selected_triangles is None:
                                self.selected_triangles = np.where(self.triangle_clusters == pid)[0]
                            else:
                                print(self.selected_triangles)
                                print(np.where(self.triangle_clusters == pid)[0])
                                self.selected_triangles = np.unique(np.concatenate((self.selected_triangles, np.where(self.triangle_clusters == pid)[0])))
                            self.left_mouse_pos = None
                        
                    elif self.right_mouse_pos is not None and self.right_mouse_pos[0] < self.W and self.right_mouse_pos[1] < self.H and self.decompose:
                        if self.surface:
                            self.selected_triangles = None
                            self.change = 3
                            self.prompt_point.append([self.right_mouse_pos[0], self.right_mouse_pos[1]])
                            self.prompt_label.append(0)
                            input_point = np.array(self.prompt_point)
                            input_label = np.array(self.prompt_label)
                            masks, scores, logits = self.predictor.predict(
                                                            point_coords=input_point,
                                                            point_labels=input_label,
                                                            multimask_output=False,)
                            index = np.argmax(scores)
                            mask = masks[index]
                            mask = np.logical_or(self.mask, mask)
                        else:
                            self.change = 4
                            # self.prompt_point.append([self.right_mouse_pos[0], self.right_mouse_pos[1]])
                            # self.prompt_label.append(1)
                            input_point = np.array([[self.right_mouse_pos[0], self.right_mouse_pos[1]]])
                            input_label = np.array([1])
                            masks, scores, logits = self.predictor.predict(
                                                            point_coords=input_point,
                                                            point_labels=input_label,
                                                            multimask_output=False,)
                            index = np.argmax(scores)
                            mask = masks[index]
                            sub_mask = copy.deepcopy(mask)
                            mask = np.logical_and(self.mask, ~mask)
                            
                            ref_mask = sub_mask.astype(np.uint8) * 255
                            kernel = np.ones((5,5), np.uint8)
                            dilated = binary_dilation(ref_mask)
                            eroded = binary_erosion(ref_mask)
                            edge = np.logical_and(dilated, np.logical_not(eroded))

                            dilated_mask = cv2.dilate(ref_mask, kernel, iterations=1).astype(bool)
                            dilated = binary_dilation(dilated_mask, iterations=5)
                            eroded = binary_erosion(dilated_mask, iterations=1)
                            dilated_edge = np.logical_and(dilated, np.logical_not(eroded))
                            dilated_edge=np.logical_and(dilated_edge, layout.reshape(self.H, self.W).astype(bool))
                            # WARNING: Due to unstable rasterization result, we implement with eroded_mask
                            erode_mask = binary_erosion(ref_mask, iterations=5)
                            new_mask = (erode_mask / 255).astype(bool)
                            seed_faces = rast[..., 3:].cpu().numpy().reshape((self.H, self.W)).astype(int)
                            seed_faces = seed_faces[new_mask]
                            seed_faces = seed_faces[seed_faces != 0]
                            seed_faces = np.unique(seed_faces) - 1
                            seed_points_indices_ori = self.triangles[seed_faces]
                            values, counts = np.unique(seed_points_indices_ori, return_counts=True)
                            unique_counts_indices = np.where(counts == 1)[0]
                            seed_points_indices_ori = np.delete(seed_points_indices_ori, unique_counts_indices)
                            seed_points_indices_ori = np.unique(seed_points_indices_ori)
                            seed_points_indices = seed_points_indices_ori
                            save_pc(self.vert[seed_points_indices], "./tmp/projection.ply")
                            
                            dilated_faces = rast[..., 3:].cpu().numpy().reshape((self.H, self.W)).astype(int)
                            dilated_faces = dilated_faces[dilated_edge]
                            dilated_faces = dilated_faces[dilated_faces != 0]
                            dilated_faces = np.unique(dilated_faces) - 1
                            dilated_points_indices_ori = self.triangles[dilated_faces]
                            values, counts = np.unique(dilated_points_indices_ori, return_counts=True)
                            unique_counts_indices = np.where(counts == 1)[0]
                            dilated_points_indices_ori = np.delete(dilated_points_indices_ori, unique_counts_indices)
                            dilated_points_indices_ori = np.unique(dilated_points_indices_ori)
                            dilated_points_indices = dilated_points_indices_ori
                            n_points = self.vert.shape[0]
                            labels = np.zeros(n_points, dtype=np.int32)
                            seed_points_indices_keep = copy.deepcopy(seed_points_indices)
                            save_pc(self.vert[seed_points_indices_keep], "./tmp/seeds.ply")
                            adj_list = copy.deepcopy(self.adj_list)
                            return_indices = surface_growing(self.vert, self.all_vert_feature.cpu(), seed_points_indices, dilated_points_indices, adj_list, self.vert, self.triangles, labels, accuracy=0.001)              
                            return_pc = self.vert[return_indices != 0]
                            vert_indices = return_indices != 0
                            if len(return_pc) < len(self.vert[seed_points_indices_keep]):
                                vert_indices = seed_points_indices_keep
                            target_vertices = np.where(vert_indices)[0]
                            triangle_vertices = self.triangles.reshape(-1)
                            triangle_index_range = np.arange(0, len(triangle_vertices), 3)
                            sub_triangle_mask = np.isin(triangle_vertices, target_vertices)
                            sub_triangle_mask = np.logical_or.reduce(
                                    (sub_triangle_mask[triangle_index_range],
                                    sub_triangle_mask[triangle_index_range+1],
                                    sub_triangle_mask[triangle_index_range+2]))
                            sub_triangles = np.where(sub_triangle_mask)[0]
                            self.selected_triangles = np.setdiff1d(self.selected_triangles, sub_triangles)
                            final_mesh = copy.deepcopy(self.fg_mesh_o3d)
                            final_mesh.remove_triangles_by_mask(~sub_triangle_mask)
                            o3d.io.write_triangle_mesh("./tmp/mesh_extracted.ply", final_mesh)
                            # self.operation = False
                            self.right_mouse_pos = None

                        
                    elif self.mouse_loc is not None and self.mouse_loc[0] < self.W and self.mouse_loc[1] < self.H and self.prompt_label == [] and self.prompt_point == []:
                        input_point = np.array([[self.mouse_loc[0], self.mouse_loc[1]]])
                        input_label = np.array([1])
                        if self.surface:
                            masks, scores, logits = self.predictor.predict(
                                                            point_coords=input_point,
                                                            point_labels=input_label,
                                                            multimask_output=False,)
                            index = np.argmax(scores)
                            mask = masks[index]
                        else:
                            if self.selected_triangles is None:
                                input_point = input_point.astype(int)
                                pid = ccid[input_point[0][1]][input_point[0][0]]
                                mask = ccid == pid
                            else:
                                masks, scores, logits = self.predictor.predict(
                                                            point_coords=input_point,
                                                            point_labels=input_label,
                                                            multimask_output=False,)
                                index = np.argmax(scores)
                                flip_mask = masks[index]
                    else:
                        if self.decompose and len(self.prompt_point) > 0 and len(self.prompt_label) > 0:
                            input_point = np.array(self.prompt_point)
                            input_label = np.array(self.prompt_label)
                            masks, scores, logits = self.predictor.predict(
                                                        point_coords=input_point,
                                                        point_labels=input_label,
                                                        multimask_output=False,)
                            index = np.argmax(scores)
                            mask = masks[index]
                            mask = np.logical_or(self.mask, mask)
                        else:
                            self.prompt_label = []
                            self.prompt_point = []
                    if self.change > 0 and self.grow:
                        self.operation = False
                        if self.change == 1 or self.change == 3 or self.change == 4:
                            if self.change == 4:
                                ref_mask = sub_mask.astype(np.uint8) * 255
                            else:
                                ref_mask = mask.astype(np.uint8) * 255
                            kernel = np.ones((5,5), np.uint8)

                            dilated = binary_dilation(ref_mask)
                            eroded = binary_erosion(ref_mask)
                            edge = np.logical_and(dilated, np.logical_not(eroded))

                            dilated_mask = cv2.dilate(ref_mask, kernel, iterations=1).astype(bool)
                            dilated = binary_dilation(dilated_mask, iterations=5)
                            eroded = binary_erosion(dilated_mask, iterations=1)
                            dilated_edge = np.logical_and(dilated, np.logical_not(eroded))
                            dilated_edge=np.logical_and(dilated_edge, layout.reshape(self.H, self.W).astype(bool))
                            # WARNING: Due to unstable rasterization result, we implement with eroded_mask
                            erode_mask = binary_erosion(ref_mask, iterations=5)
                            new_mask = (erode_mask / 255).astype(bool)
                            seed_faces = rast[..., 3:].cpu().numpy().reshape((self.H, self.W)).astype(int)
                            seed_faces = seed_faces[new_mask]
                            seed_faces = seed_faces[seed_faces != 0]
                            seed_faces = np.unique(seed_faces) - 1
                            seed_points_indices_ori = self.triangles[seed_faces]
                            values, counts = np.unique(seed_points_indices_ori, return_counts=True)
                            unique_counts_indices = np.where(counts == 1)[0]
                            seed_points_indices_ori = np.delete(seed_points_indices_ori, unique_counts_indices)
                            seed_points_indices_ori = np.unique(seed_points_indices_ori)
                            seed_points_indices = seed_points_indices_ori
                            save_pc(self.vert[seed_points_indices], "./tmp/projection.ply")
                            
                            dilated_faces = rast[..., 3:].cpu().numpy().reshape((self.H, self.W)).astype(int)
                            dilated_faces = dilated_faces[dilated_edge]
                            dilated_faces = dilated_faces[dilated_faces != 0]
                            dilated_faces = np.unique(dilated_faces) - 1
                            dilated_points_indices_ori = self.triangles[dilated_faces]
                            values, counts = np.unique(dilated_points_indices_ori, return_counts=True)
                            unique_counts_indices = np.where(counts == 1)[0]
                            dilated_points_indices_ori = np.delete(dilated_points_indices_ori, unique_counts_indices)
                            dilated_points_indices_ori = np.unique(dilated_points_indices_ori)
                            dilated_points_indices = dilated_points_indices_ori

                            # check_edge = dilated_edge[1:-1,1:-1]
                            # layout_cropped = layout[1:-1,1:-1]
                            # x = layout_cropped[check_edge]
                            # independent_cluster = False
                            # if len(x) == 0:
                                # independent_cluster = True
                            # # Determine the cluster index
                            # indice_check = np.isin(self.triangles, seed_points_indices)
                            # tri_idx, _ = np.where(indice_check)
                            # counter = Counter(self.triangle_clusters[tri_idx])
                            # most_common_element, count = counter.most_common(1)[0]
                            # cluster_idx = most_common_element #processed_triangle_clusters[tri_idx]
                            # all_cluster_idx = np.unique(self.triangle_clusters[tri_idx])
                            
                            # if not independent_cluster:
                            n_points = self.vert.shape[0]
                            labels = np.zeros(n_points, dtype=np.int32)
                            seed_points_indices_keep = copy.deepcopy(seed_points_indices)
                            save_pc(self.vert[seed_points_indices_keep], "./tmp/seeds.ply")
                            adj_list = copy.deepcopy(self.adj_list)
                            return_indices = surface_growing(self.vert, self.all_vert_feature.cpu(), seed_points_indices, dilated_points_indices, adj_list, self.vert, self.triangles, labels, accuracy=0.001)              
                            return_pc = self.vert[return_indices != 0]
                            vert_indices = return_indices != 0
                            if len(return_pc) < len(self.vert[seed_points_indices_keep]):
                                vert_indices = seed_points_indices_keep
                            # if independent_cluster:
                            #     if len(all_cluster_idx) == 1:
                            #         result_mesh = o3d.io.read_triangle_mesh("./tmp/raw_decomposition/"+str(cluster_idx)+".ply")
                            #         o3d.io.write_triangle_mesh("./tmp/mesh_extracted.ply", result_mesh)
                            #         self.selected_triangles = np.where(self.triangle_clusters == cluster_idx)[0]
                            #     else:
                            #         mesh_component = copy.deepcopy(self.fg_mesh)
                            #         triangles_to_remove = ~np.in1d(self.triangle_clusters, all_cluster_idx).reshape(self.triangle_clusters.shape)
                            #         self.selected_triangles = self.triangle_clusters[np.in1d(self.triangle_clusters, all_cluster_idx).reshape(self.triangle_clusters.shape)]
                            #         mesh_component.remove_triangles_by_mask(triangles_to_remove) 
                            #         o3d.io.write_triangle_mesh("./tmp/mesh_extracted.ply", mesh_component) 
                            # else:                  
                            target_vertices = np.where(vert_indices)[0]
                            triangle_vertices = self.triangles.reshape(-1)
                            triangle_index_range = np.arange(0, len(triangle_vertices), 3)
                            sub_triangle_mask = np.isin(triangle_vertices, target_vertices)
                            sub_triangle_mask = np.logical_or.reduce(
                                    (sub_triangle_mask[triangle_index_range],
                                    sub_triangle_mask[triangle_index_range+1],
                                    sub_triangle_mask[triangle_index_range+2]))
                            sub_triangles = np.where(sub_triangle_mask)[0]
                            if self.change == 4:
                                self.selected_triangles = np.setdiff1d(self.selected_triangles, sub_triangles)
                            else:
                                self.selected_triangles = sub_triangles
                            final_mesh = copy.deepcopy(self.fg_mesh_o3d)
                            final_mesh.remove_triangles_by_mask(~sub_triangle_mask)
                            o3d.io.write_triangle_mesh("./tmp/mesh_extracted.ply", final_mesh)
                        else:
                            mesh_component = copy.deepcopy(self.fg_mesh)
                            clusters = np.array(self.clusters)
                            triangles_to_remove = ~np.in1d(self.triangle_clusters, clusters).reshape(self.triangle_clusters.shape)
                            self.selected_triangles = self.triangle_clusters[np.in1d(self.triangle_clusters, clusters).reshape(self.triangle_clusters.shape)]
                            mesh_component.remove_triangles_by_mask(triangles_to_remove) 
                            o3d.io.write_triangle_mesh("./tmp/mesh_extracted.ply", mesh_component) 
                            
                        self.change = 0
                        self.grow = False
                    # if self.selected_triangles is not None:
                    #     mask = np.in1d(faces.reshape(self.H, self.W), self.selected_triangles)
                    # For demo
                    color = np.array([30/255, 144/255, 255/255])
                    if self.selected_triangles is None:               
                        if np.any(self.mask) or len(self.prompt_point) > 0:
                            kernel = np.ones((10, 10), np.uint8)
                            inside_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
                            display_mask = cv2.bitwise_xor(mask.astype(np.uint8), inside_mask)
                            outside_mask = ~mask # cv2.bitwise_or(cv2.bitwise_not(mask.astype(np.uint8)), display_mask)
                            display_mask = display_mask.reshape(self.H, self.W).astype(bool)
                            outside_mask = outside_mask.reshape(self.H, self.W).astype(bool)
                            # kernel = np.ones((10, 10), np.uint8)
                            # inside_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
                            # display_mask = np.logical_xor(mask, inside_mask)
                            # outside_mask = np.logical_or(~mask, display_mask)
                            mask_image = display_mask.reshape(1, self.H, self.W) * color.reshape(-1, 1, 1)
                            count = 0
                            for p in self.prompt_point:
                                x, y = p
                                radius = 8
                                rr, cc = np.ogrid[:self.H, :self.W]
                                circle = (rr - y) ** 2 + (cc - x) ** 2 <= radius ** 2
                                mask_image = np.transpose(mask_image, (1,2,0))
                                if self.prompt_label[count] == 1:
                                    mask_image[circle] = np.array([1,1,1])
                                else:
                                    mask_image[circle] = np.array([1,0,0])
                                mask_image = np.transpose(mask_image, (2,0,1))
                                count += 1
                            mask_image = torch.from_numpy(mask_image.astype(np.float32)).to(self.device)
                            buffer_image = buffer_image.transpose(0,2).transpose(0,1)
                            buffer_image[outside_mask] = buffer_image[outside_mask] * (1 - self.overlay_input_img_ratio)
                            buffer_image[display_mask] = buffer_image[display_mask] * (1 - self.overlay_input_img_ratio)
                            buffer_image = buffer_image.transpose(0,2).transpose(1,2)
                            buffer_image = buffer_image + mask_image * self.overlay_input_img_ratio
                            # buffer_image = buffer_image * (1 - self.overlay_input_img_ratio) \
                            #             + mask_image * self.overlay_input_img_ratio
                        else:
                            mask_image = mask.reshape(1, self.H, self.W) * color.reshape(-1, 1, 1)
                            mask_image = torch.from_numpy(mask_image.astype(np.float32)).to(self.device)
                            buffer_image = buffer_image.transpose(0,2).transpose(0,1)
                            buffer_image[mask] = buffer_image[mask] * (1 - self.overlay_input_img_ratio)
                            buffer_image = buffer_image.transpose(0,2).transpose(1,2)
                            buffer_image = buffer_image + mask_image * self.overlay_input_img_ratio            
                        self.mouse_loc = None
                        self.left_mouse_pos = None
                        self.right_mouse_pos = None
                        self.history.append(self.selected_triangles)
                    else:
                        mask = np.in1d(faces.reshape(self.H, self.W), self.selected_triangles)
                        if np.any(flip_mask):
                            mask = np.logical_and(mask.reshape(self.H, self.W), ~flip_mask)
                        color = np.array([30/255, 144/255, 255/255])
                        mask_image = mask.reshape(1, self.H, self.W) * color.reshape(-1, 1, 1)
                        mask_image = torch.from_numpy(mask_image.astype(np.float32)).to(self.device)
                        buffer_image = buffer_image * (1 - self.overlay_input_img_ratio) \
                                        + mask_image * self.overlay_input_img_ratio 
                else:
                    pose = torch.from_numpy(self.cam.pose_mesh.astype(np.float32)).cuda()
                    proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).cuda()
                    v_cam = torch.matmul(F.pad(self.fg_mesh.v, pad=(0, 1), mode='constant', value=1.0), 
                                        torch.inverse(pose).T).float().unsqueeze(0)
                    v_clip = v_cam @ proj.T
                    v_clip[..., :1] -= 0.5 / self.H
                    v_clip[..., 1:2] += 0.5 / self.H

                    rast, rast_db = dr.rasterize(self.glctx, v_clip, self.fg_mesh.f, (self.H, self.W))
                    alpha = (rast[..., 3:] > 0).float()
                    faces = rast[..., 3:].cpu().numpy().reshape(-1).astype(int) - 1
                    faces[np.where(faces == -1)[0]] = len(self.triangle_clusters) + 10
                    mask = np.zeros((self.H, self.W)).astype(bool)
                    if self.selected_triangles is not None:
                        mask = np.in1d(faces.reshape(self.H, self.W), self.selected_triangles)
                        color = np.array([30/255, 144/255, 255/255])
                        mask_image = mask.reshape(1, self.H, self.W) * color.reshape(-1, 1, 1)
                        mask_image = torch.from_numpy(mask_image.astype(np.float32)).to(self.device)
                        buffer_image = buffer_image * (1 - self.overlay_input_img_ratio) \
                                        + mask_image * self.overlay_input_img_ratio 
                self.buffer_image = (
                    buffer_image.permute(1, 2, 0)
                    .contiguous()
                    .clamp(0, 1)
                    .contiguous()
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                buffer_image = out[self.mode]  # [3, H, W]
                if self.mode in ['depth', 'alpha']:
                    buffer_image = buffer_image.repeat(3, 1, 1)
                    if self.mode == 'depth':
                        buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)
                        
                self.buffer_image = (
                    buffer_image.permute(1, 2, 0)
                    .contiguous()
                    .clamp(0, 1)
                    .contiguous()
                    .detach()
                    .cpu()
                    .numpy()
                )
            self.need_update = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:
            if self.mode == "decompose" and self.operation:
                dpg.set_value("_log_infer_time", "--")
            else:
                dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", self.buffer_image
            )  # buffer must be contiguous, else seg fault!

    
    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024):
        os.makedirs(self.opt.outdir, exist_ok=True)
        if mode == 'geo':
            # NOTE: We don't need to directly save the reconstructed mesh here.
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.ply')
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)
            mesh.write_ply(path)

        else:
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')
            self.renderer.gaussians.save_ply(path)

        print(f"[INFO] save model to {path}.")

    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # input stuff
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        dpg.set_value("_log_input", k)
                        self.load_input(v)

                    self.need_update = True

                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=callback_select_input,
                    file_count=1,
                    tag="file_dialog_tag",
                    width=700,
                    height=400,
                ):
                    dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="input",
                        callback=lambda: dpg.show_item("file_dialog_tag"),
                    )
                    dpg.add_text("", tag="_log_input")
                
                # overlay stuff
                with dpg.group(horizontal=True):

                    def callback_toggle_overlay_input_img(sender, app_data):
                        self.overlay_input_img = not self.overlay_input_img
                        self.need_update = True

                    dpg.add_checkbox(
                        label="overlay image",
                        default_value=self.overlay_input_img,
                        callback=callback_toggle_overlay_input_img,
                    )

                    def callback_set_overlay_input_img_ratio(sender, app_data):
                        self.overlay_input_img_ratio = app_data
                        self.need_update = True

                    dpg.add_slider_float(
                        label="ratio",
                        min_value=0,
                        max_value=1,
                        format="%.1f",
                        default_value=self.overlay_input_img_ratio,
                        callback=callback_set_overlay_input_img_ratio,
                    )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    def callback_save(sender, app_data, user_data):
                        self.save_model(mode=user_data)

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=callback_save,
                        user_data='model',
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            self.prepare_train()
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")

                    # dpg.add_button(
                    #     label="init", tag="_button_init", callback=self.prepare_train
                    # )
                    # dpg.bind_item_theme("_button_init", theme_button)

                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_time")
                    dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    # ("image", "sam", "mesh", "depth", "alpha", "layout", "decompose"),
                    ("image", "mesh", "depth", "alpha", "layout", "decompose"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )

                def callback_set_gaussain_scale(sender, app_data):
                    self.gaussain_scale_factor = app_data
                    self.need_update = True

                dpg.add_slider_float(
                    label="gaussain scale",
                    min_value=0,
                    max_value=1,
                    format="%.2f",
                    default_value=self.gaussain_scale_factor,
                    callback=callback_set_gaussain_scale,
                )

            # light dir
            with dpg.collapsing_header(label="Lighting Setting (Only work for mesh mode)", default_open=True):
                def callback_set_light_dir(sender, app_data, user_data):
                    self.light_dir[user_data] = app_data
                    self.need_update = True

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="theta", min_value=0, max_value=180, 
                                         format="%.2f", default_value=self.light_dir[0], 
                                         callback=callback_set_light_dir, user_data=0)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="phi", min_value=0, max_value=360, 
                                         format="%.2f", default_value=self.light_dir[1], 
                                         callback=callback_set_light_dir, user_data=1)
            # Selection dir
            with dpg.collapsing_header(label="Decomposition Settings (Only work for decompose mode)", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_text("Current Mode: ")
                    
                    def callback_view_mode(sender, app_data):
                        self.operation = False
                        self.need_update = True
                        dpg.configure_item("_button_view_mode", label="view")
                    dpg.add_button(
                        label="view", tag="_button_view_mode", callback=callback_view_mode
                    )
                    dpg.bind_item_theme("_button_view_mode", theme_button)
                    
                    def callback_operation_mode(sender, app_data):
                        self.operation = True
                        self.need_update = True
                        dpg.configure_item("_button_operation_mode", label="operation")
                    dpg.add_button(
                        label="operation", tag="_button_operation_mode", callback=callback_operation_mode
                    )
                    dpg.bind_item_theme("_button_operation_mode", theme_button)
                    
                with dpg.group(horizontal=True):
                    dpg.add_text("Selection Strategy: ")
                    
                    def callback_selection_seed_vertices(sender, app_data):
                        self.surface = True
                        self.need_update = True
                        dpg.configure_item("_button_selection_sv", label="seed vertices")
                    dpg.add_button(
                        label="seed vertices", tag="_button_selection_sv", callback=callback_selection_seed_vertices
                    )
                    
                    def callback_selection_connected_component(sender, app_data):
                        self.surface = False
                        self.need_update = True
                        dpg.configure_item("_button_selection_cc", label="connected component")

                    dpg.add_button(
                        label="connected component", tag="_button_selection_cc", callback=callback_selection_connected_component
                    )
                    
                    dpg.bind_item_theme("_button_selection_sv", theme_button)
                    dpg.bind_item_theme("_button_selection_cc", theme_button)
                    
                    # dpg.add_text("Selection mode: ")
                with dpg.group(horizontal=True):
                    dpg.add_text("Operations: ")
                    
                    def callback_selection(sender, app_data):
                        self.grow = True
                        self.need_update = True
                        dpg.configure_item("_button_grow", label="grow")
                    dpg.add_button(
                        label="grow", tag="_button_grow", callback=callback_selection
                    )
                    dpg.bind_item_theme("_button_grow", theme_button)
                    
                    def callback_undo(sender, app_data):
                        self.undo = True
                        self.need_update = True
                        dpg.configure_item("_button_undo", label="undo")
                    dpg.add_button(
                        label="undo", tag="_button_undo", callback=callback_undo
                    )
                    dpg.bind_item_theme("_button_undo", theme_button)
                    
                    def callback_reset(sender, app_data):
                        self.history = []
                        self.prompt_point = []
                        self.prompt_label = []
                        self.mask = np.zeros((self.H, self.W)).astype(bool)
                        self.decompose = False
                        self.need_update = True
                        self.selected_triangles = None
                        dpg.configure_item("_button_reset", label="reset")
                    dpg.add_button(
                        label="reset", tag="_button_reset", callback=callback_reset
                    )
                    dpg.bind_item_theme("_button_reset", theme_button)

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True
        
        # def callback_right_mouse_double_clicked(sender, app_data):
        #     if not dpg.is_item_focused("_primary_window"):
        #         return

        #     self.undo = dpg.is_mouse_button_double_clicked(dpg.mvMouseButton_Right)
        #     self.need_update = True
            
        # def callback_left_mouse_double_clicked(sender, app_data):
        #     if not dpg.is_item_focused("_primary_window"):
        #         return

        #     self.decompose = dpg.is_mouse_button_double_clicked(dpg.mvMouseButton_Left)
        #     self.need_update = True
        def callback_space_pressed(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            if dpg.is_key_pressed(dpg.mvKey_Spacebar):
                self.grow = True
                
            self.need_update = True
        
        def callback_shift_pressed(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            if dpg.is_key_pressed(dpg.mvKey_Shift):
                self.decompose = True
                self.mouse_pos = None
                self.finish = False
                self.extracted = None
                
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = dpg.get_mouse_pos()
            self.mouse_loc[0] -= 1
            self.mouse_loc[1] += 17
            self.need_update = True

        def callback_set_mouse_left_pos(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            # just the pixel coordinate in image
            self.left_mouse_pos = dpg.get_mouse_pos()
            self.left_mouse_pos[0] -= 1
            self.left_mouse_pos[1] += 17
            self.need_update = True
            
        def callback_set_mouse_right_pos(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            # just the pixel coordinate in image
            self.right_mouse_pos = dpg.get_mouse_pos()
            self.right_mouse_pos[0] -= 1
            self.right_mouse_pos[1] += 17
            self.need_update = True
            
            
        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )
            dpg.add_mouse_click_handler(
                button=dpg.mvMouseButton_Right, callback=callback_set_mouse_right_pos
            )
            dpg.add_mouse_click_handler(
                button=dpg.mvMouseButton_Left, callback=callback_set_mouse_left_pos
            )
            dpg.add_mouse_move_handler(
                callback=callback_set_mouse_loc
            )
            # for operations related to decomposition
            # dpg.add_mouse_click_handler(
            #     button=dpg.mvMouseButton_Left, callback=callback_left_mouse_double_clicked
            # )
            # dpg.add_mouse_hover_handler(
            #     callback=callback_set_mouse_loc
            # )
            # dpg.add_item_hover_handler(
            #     parent= "_primary_window", callback=callback_set_mouse_loc
            # )
            dpg.add_key_press_handler(
                key=dpg.mvKey_Shift, callback=callback_shift_pressed
            )
            dpg.add_key_press_handler(
                key=dpg.mvKey_Spacebar, callback=callback_space_pressed
            )

        dpg.create_viewport(
            title="Total-decom",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()
    
    # no gui mode
    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step()
            # do a last prune
            self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
        # save
        self.save_model(mode='model')
        self.save_model(mode='geo+tex')
        

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    if not torch.cuda.is_available():
        raise RuntimeError("The Total-Decom GUI requires a CUDA-capable NVIDIA GPU.")
    torch.cuda.set_device(int(opt.get("device_index", 0)))

    # TODO: Consider the inference cameras, and how to define them

    gui = GUI(opt)

    if opt.gui:
        gui.render()
    else:
        gui.train(opt.epochs)
