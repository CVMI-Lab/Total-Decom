import os
import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append(".")
from rend_util import load_K_Rt_from_P
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_dir", type=str, required=True)
    parser.add_argument("-m", "--mesh_path", type=str, required=True)
    parser.add_argument("-s", "--saved_path", type=str)
    args = parser.parse_args()
    return args

class Renderer():
    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        # self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)  # , self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()

if __name__ == "__main__":
    args = parse_args()
    mesh = trimesh.load(args.mesh_path)
    cameras_dict = np.load(os.path.join(args.data_dir, "cameras.npz"))
    saved_path = args.saved_path
    os.makedirs(saved_path, exist_ok=True)

    H, W = 384, 384
    renderer = Renderer()
    for i in tqdm(range(len(cameras_dict) // 2)):
        depth_path = os.path.join(saved_path, "{:06d}_bg_depth.npy".format(i))
        depth_vis_path = os.path.join(saved_path, "{:06d}_bg_depth.png".format(i))
        scale_mat = cameras_dict['scale_mat_{}'.format(i)]
        world_mat = cameras_dict['world_mat_{}'.format(i)]
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsic, pose = load_K_Rt_from_P(None, P)
        intrinsic = intrinsic[:3, :3]
        
        mesh_opengl = renderer.mesh_opengl(mesh)
        _, depth_pred = renderer(H, W, intrinsic, pose, mesh_opengl)
        np.save(depth_path, depth_pred)
        plt.imshow(depth_pred)
        plt.axis("off")
        plt.savefig(depth_vis_path)
