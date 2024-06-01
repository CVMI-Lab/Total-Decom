import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import rend_util
from model.embedder import *
from model.density import LaplaceDensity
from model.ray_sampler import ErrorBoundSampler, UniformSampler
import matplotlib.pyplot as plt
import numpy as np

import open3d as o3d

from hashencoder.hashgrid import _hash_encode, HashEncoder

class SemNetwork(nn.Module):
    def __init__(
        self,
        din,
        label_dim,
        weight_norm=True,
    ):
        super().__init__()
        dims = [din, din // 2, label_dim]
        self.num_layers = len(dims)
        for l in range(0, len(dims) - 1):
            lin = nn.Linear(dims[l], dims[l + 1])
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.relu = nn.ReLU()

    def forward(self, x):
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
        return x

class SemImplicitGrid(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            sigmoid=20, # NOTE: Check the function of this param
            inside_outside=False,
            base_size = 16,
            end_size = 2048,
            logmap = 19,
            num_levels=16,
            level_dim=2,
            divide_factor = 1.5, # used to normalize the points range for multi-res grid
            use_grid_feature = True
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        self.divide_factor = divide_factor
        self.grid_feature_dim = num_levels * level_dim
        self.use_grid_feature = use_grid_feature

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.d_out = d_out
        self.sigmoid = sigmoid
        
        dims[0] += self.grid_feature_dim
        print(f"using hash encoder with {num_levels} levels, each level with feature dim {level_dim}")
        print(f"resolution:{base_size} -> {end_size} with hash map size {logmap}")
        self.encoding = HashEncoder(input_dim=3, num_levels=num_levels, level_dim=level_dim, 
                    per_level_scale=2, base_resolution=base_size, 
                    log2_hashmap_size=logmap, desired_resolution=end_size)

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] += input_ch - 3
        print("network architecture")
        print(dims)
        
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    # NOTE: Here we change the initialize method follow the objsdf++
                    # if not inside_outside:
                    #     torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    #     torch.nn.init.constant_(lin.bias, -bias)
                    # else:
                    #     torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    #     torch.nn.init.constant_(lin.bias, bias)
                    
                    # Geometry initalization for compositional scene, bg SDF sign: inside + outside -, fg SDF sign: outside + inside -
                    # The 0 index is the background SDF, the rest are the object SDFs
                    # background SDF with postive value inside and nagative value outside
                    torch.nn.init.normal_(lin.weight[:1, :], mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias[:1], -0.5*bias)
                    # inner objects with SDF initial with negative value inside and positive value outside, ~0.5 radius of background
                    torch.nn.init.normal_(lin.weight[1:,:], mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias[1:], bias)

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.pool = nn.MaxPool1d(self.d_out)
        self.cache_sdf = None


    def forward(self, input):
        if self.use_grid_feature:
            # normalize point range as encoding assume points are in [-1, 1]
            feature = self.encoding(input / self.divide_factor)
        else:
            feature = torch.zeros_like(input[:, :1].repeat(1, self.grid_feature_dim))
                    
        if self.embed_fn is not None:
            embed = self.embed_fn(input)
            input = torch.cat((embed, feature), dim=-1)
        else:
            input = torch.cat((input, feature), dim=-1)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:self.d_out]
        d_output = torch.ones_like(y[:, :1], requires_grad=False, device=y.device)
        f = lambda v: torch.autograd.grad(outputs=y,
                    inputs=x,
                    grad_outputs=v.repeat(y.shape[0], 1),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
        
        N = torch.eye(y.shape[1], requires_grad=False).to(y.device)
        
        # start_time = time.time()
        g = torch.cat([torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=idx.repeat(y.shape[0], 1),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0] for idx in N.unbind()])
        # torch.cuda.synchronize()
        # print("time for computing gradient by for loop: ", time.time() - start_time, "s")
        
        sdf = -self.pool(-y.unsqueeze(1)).squeeze(-1)
        g_min_sdf = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        g = torch.cat([g, g_min_sdf])
        return g

    # def gradient(self, x):
    #     x.requires_grad_(True)
    #     y = self.forward(x)[:,:self.d_out]
    #     d_output = torch.ones_like(y[:, :1], requires_grad=False, device=y.device)
    #     g = []
    #     for idx in range(y.shape[1]):
    #         gradients = torch.autograd.grad(
    #             outputs=y[:, idx:idx+1],
    #             inputs=x,
    #             grad_outputs=d_output,
    #             create_graph=True,
    #             retain_graph=True,
    #             only_inputs=True)[0]
    #         g.append(gradients)
        
    #     g = torch.cat(g)
    #     # print(gradients.shape)
    #     # change it to support each sdf to satisfied eki loss
    #     return g

    def get_outputs(self, x, only_layout=False):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:,:self.d_out]
        layout = self.sigmoid * torch.sigmoid(-self.sigmoid * sdf_raw)
        if only_layout:
            return layout
        sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1) # get the minium value of sdf
        # apply bound after min
        # if self.sdf_bounding_sphere > 0.0:
        #     sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
        #     sdf = torch.minimum(sdf, sphere_sdf)
        feature_vectors = output[:, self.d_out:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return sdf, feature_vectors, gradients, layout, sdf_raw

    def get_specific_outputs(self, x, idx):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:,:self.d_out]
        layout = self.sigmoid * torch.sigmoid(-self.sigmoid * sdf_raw)
        # sdf = output[:, idx:idx+1]
        sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1)
        # sdf = -self.pool(-sdf_raw) # get the minium value of sdf
        # ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        # if self.sdf_bounding_sphere > 0.0:
        #     sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
        #     sdf = torch.minimum(sdf, sphere_sdf)
        feature_vectors = output[:, self.d_out:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return sdf, feature_vectors, gradients, layout, output[:,:self.d_out]

    def get_sdf_raw(self, x):
        return self.forward(x)[:, :self.d_out]

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[:,:self.d_out]
        sdf = -self.pool(-sdf.unsqueeze(1)).squeeze(-1) # get the minium value of sdf if bound apply before min
        return sdf

    def get_specific_sdf_vals(self, x, idx, return_feature=False):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf = output[:,:self.d_out]
        # if self.sdf_bounding_sphere > 0.0:
        #     sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2, 1, keepdim=True))
        #     sdf = torch.minimum(sdf, sphere_sdf.expand(sdf.shape))
        sdf = sdf[:, idx: idx+1]
        if return_feature:
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            return sdf, output[:, self.d_out:], gradients
        else:
            return sdf

    def grid_parameters(self):
        print("grid parameters", len(list(self.encoding.parameters())))
        for p in self.encoding.parameters():
            print(p.shape)
        return self.encoding.parameters()

    def mlp_parameters(self):
        parameters = []
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            parameters += list(lin.parameters())
        return parameters

class FeatureRenderingNetwork(nn.Module):
    def __init__(self, 
                 d_in,
                 d_out,
                 dims,
                 weight_norm=True,
                 multires=0,
                 divide_factor=1.0,):
        super().__init__()
        L = 16
        F = 2
        log2_T = 19 
        N_min = 16
        self.divide_factor = divide_factor
        dims = [d_in] + dims + [d_out]
        self.embed_fn = None
        self.grid_feature_dim = L * F
        dims[0] += self.grid_feature_dim
        self.encoding = HashEncoder(input_dim=3, num_levels=L, 
                                    level_dim=F, per_level_scale=2, 
                                    base_resolution=N_min, log2_hashmap_size=log2_T, 
                                    desired_resolution=128)
        
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] += input_ch - 3
        print("network architecture")
        print(dims)
        
        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

    def forward(self, input):

        # We assume the feature rendering network always uses the hash grid representation       
        feature = self.encoding(input / self.divide_factor)

        if self.embed_fn is not None:
            embed = self.embed_fn(input)
            input = torch.cat((embed, feature), dim=-1)
        else:
            input = torch.cat((input, feature), dim=-1)

        x = input
        
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l < self.num_layers - 2:
                x = torch.relu(lin(x))
            else:
                x = lin(x)

        return x

class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
            per_image_code = False
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.per_image_code = per_image_code
        if self.per_image_code:
            # nerf in the wild parameter
            # parameters
            # maximum 1024 images
            self.embeddings = nn.Parameter(torch.empty(1024, 32))
            std = 1e-4
            self.embeddings.data.uniform_(-std, std)
            dims[0] += 32

        print("rendering network architecture:")
        print(dims)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, normals, view_dirs, feature_vectors, indices):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        else:
            raise NotImplementedError

        if self.per_image_code:
            image_code = self.embeddings[indices].expand(rendering_input.shape[0], -1)
            rendering_input = torch.cat([rendering_input, image_code], dim=-1)
            
        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
        
        x = self.sigmoid(x)
        return x


# Delete the original monosdf settings
class SemMonoSDFNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)
        self.white_bkgd = conf.get_bool('white_bkgd', default=False)
        self.bg_color = torch.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0])).float().cuda()

        self.Grid_MLP = conf.get_bool('Grid_MLP', default=False)
        self.implicit_network = SemImplicitGrid(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        # NOTE: The semantic network here only focus to the wall, ceiling, floor and others
        self.sem_network = SemNetwork(**conf.get_config('sem_network'))

        self.feature_network = FeatureRenderingNetwork(**conf.get_config('feature_network'))

        self.density = LaplaceDensity(**conf.get_config('density'))
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))

        self.num_layout = conf.get_int('implicit_network.d_out')
        self.num_semantic = conf.get_int('sem_network.label_dim')
    def forward(self, input_dict, indices):
        # Parse model input
        intrinsics = input_dict["intrinsics"]
        uv = input_dict["uv"]
        pose = input_dict["pose"]

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        ray_dirs_tmp, _ = rend_util.get_camera_params(uv, torch.eye(4).to(pose.device)[None], intrinsics)
        depth_scale = ray_dirs_tmp[0, :, 2:]

        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        # Only sample the z_vals for the background field
        z_vals_bg, _ = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self, bg_field=True)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)
        
        # Sample the bg points
        points_bg = cam_loc.unsqueeze(1) + z_vals_bg.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat_bg = points_bg.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        dirs_flat = dirs.reshape(-1, 3)
        
        # Here the 0 is represented the background, and the 1 is represented the foreground
        sdf, feature_vectors, gradients, layout, sdf_raw = self.implicit_network.get_outputs(points_flat)
        sdf_bg, feature_vectors_bg, gradients_bg = self.implicit_network.get_specific_sdf_vals(points_flat_bg, 
                                                                                                1, 
                                                                                                return_feature=True)

        rendering_feature = self.feature_network(points_flat)
        rendering_feature = rendering_feature.reshape(-1, N_samples, self.feature_vector_size)

        # Using the seperately semantic network to estimate the semantic
        # shape of semantic is [batch_size * num_pixels, num_semantic]
        # semantic = self.sem_network(feature_vectors.detach())
        # semantic = semantic.reshape(-1, N_samples, self.num_semantic)

        semantic = self.sem_network(feature_vectors_bg.detach())
        semantic = semantic.reshape(-1, N_samples, self.num_semantic)

        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, indices)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        layout = layout.reshape(-1, N_samples, self.num_layout)

        weights, transmittance, dists = self.volume_rendering(z_vals, sdf)
        weights_bg, transmittance_bg, dists_bg = self.volume_rendering(z_vals_bg, sdf_bg)

        object_opacity = self.occlusion_opacity(z_vals, transmittance, dists, sdf_raw).sum(-1).transpose(0, 1)
        
        # To make sure the background opacity is 1
        object_opacity_bg = self.occlusion_opacity(z_vals, transmittance_bg, dists_bg, sdf_bg).sum(-1).transpose(0, 1)
        object_opacity_fg = torch.ones_like(object_opacity_bg)
        object_opacity_bg = torch.cat([object_opacity_fg, object_opacity_bg], dim=1)

        semantic_values = torch.sum(weights_bg.unsqueeze(-1).detach() * semantic, 1)
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        layout_values = torch.sum(weights.unsqueeze(-1).detach() * layout, 1)
        feature_values = torch.sum(weights.unsqueeze(-1).detach() * rendering_feature, 1)

        # depth values
        depth_values = torch.sum(weights * z_vals, 1, keepdims=True) / (weights.sum(dim=1, keepdims=True) +1e-8)
        depth_values = depth_scale * depth_values

        hit_points = cam_loc + depth_values * ray_dirs
        hit_dirs = torch.tensor([0, 0, -1], device=hit_points.device).reshape(1, 3)
        points_floor, mask = self.ray_marching_surface(hit_points, hit_dirs.repeat(hit_points.shape[0], 1))
        if mask.sum() == 0:
            normals_floor = None
        else:
            _, _, gradients_floor = self.implicit_network.get_specific_sdf_vals(points_floor, 1, return_feature=True)
            normals_floor = gradients_floor / (gradients_floor.norm(2, -1, keepdim=True) + 1e-6)        

        # depth_values_bg = torch.sum(weights_bg * z_vals_bg, 1, keepdims=True) / (weights_bg.sum(dim=1, keepdims=True) +1e-8)
        # depth_values_bg = depth_scale * depth_values_bg

        output = {
            'rgb':rgb,
            'rgb_values': rgb_values,
            'depth_values': depth_values,
            'normals_floor': normals_floor,
            # 'depth_values_bg': depth_values_bg,
            'object_opacity': object_opacity,
            'object_opacity_bg': object_opacity_bg,
            'z_vals': z_vals,
            'depth_vals': z_vals * depth_scale,
            'sdf': sdf.reshape(z_vals.shape),
            'weights': weights,
            'layout_probs': layout_values,
            'sem_probs': semantic_values,
            'feature_values': feature_values,
        }

        if self.training:
            # Sample points for the eikonal loss
            n_eik_points = batch_size * num_pixels
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere).cuda()

            # add some of the near surface points
            eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)

            neighbour_points = eikonal_points + (torch.rand_like(eikonal_points) - 0.5) * 0.01   
            eikonal_points = torch.cat([eikonal_points, neighbour_points], 0)
                   
            # Here the grad_theta get the sdf value for every sdf field
            grad_theta = self.implicit_network.gradient(eikonal_points)

            sample_sdf = self.implicit_network.get_sdf_raw(eikonal_points)
            sdf_value = self.implicit_network.get_sdf_vals(eikonal_points)
            output['sample_sdf'] = sample_sdf
            output['sample_minsdf'] = sdf_value
            
            # split gradient to eikonal points and heighbour ponits
            output['grad_theta'] = grad_theta[:grad_theta.shape[0]//2]
            output['grad_theta_nei'] = grad_theta[grad_theta.shape[0]//2:]

            with torch.no_grad():
                sdf_bg_reshape = sdf_bg.reshape(-1, N_samples)
                surfaces_indices = self.get_bg_surface(sdf_bg_reshape)
                # Generate the mask have true value larger than the surface indices
                bg_surface_mask = torch.arange(N_samples).expand(sdf_bg_reshape.shape[0], -1).cuda() > surfaces_indices.unsqueeze(-1)


            # Get the foreground sdf value from the background sampling points
            sdf_fg, _, _ = self.implicit_network.get_specific_sdf_vals(points_flat_bg, 0, return_feature=True)
            output['sdf_fg'] = sdf_fg.reshape(-1, N_samples)
            output['sdf_bg'] = sdf_bg.reshape(-1, N_samples)
            output['bg_surface_mask'] = bg_surface_mask

        normals_bg = gradients_bg / (gradients_bg.norm(2, -1, keepdim=True) + 1e-6)        
        normals_bg = normals_bg.reshape(-1, N_samples, 3)
        normals_bg_map = torch.sum(weights_bg.unsqueeze(-1) * normals_bg, 1)
        output['normal_bg_map_abs'] = normals_bg_map

        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)
        
        # transform to local coordinate system
        rot = pose[0, :3, :3].permute(1, 0).contiguous()
        normal_map = rot @ normal_map.permute(1, 0)
        normal_map = normal_map.permute(1, 0).contiguous()

        normal_map_bg = rot @ normals_bg_map.permute(1, 0)
        normal_map_bg = normal_map_bg.permute(1, 0).contiguous()
        
        output['normal_map'] = normal_map
        output['normal_map_bg'] = normal_map_bg

        return output

    def volume_rendering(self, z_vals, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * transmittance # probability of the ray hits something here

        return weights, transmittance, dists

    def occlusion_opacity(self, z_vals, transmittance, dists, sdf_raw):
        obj_density = self.density(sdf_raw).transpose(0, 1).reshape(-1, dists.shape[0], dists.shape[1]) # [#object, #ray, #sample points]       
        free_energy = dists * obj_density
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        object_weight = alpha * transmittance
        return object_weight


    def ray_marching_surface(self, cam_loc, ray_dirs):
        '''
        cam_loc: [N, 3]
        ray_dirs: [N, 3]
        According to indoor scene initialization, sdf should be first > 0 then < 0
        val: [N, N_samples]
        '''
        
        z_vals, near, far = self.ray_sampler.uniform_sampler.get_z_vals(ray_dirs, cam_loc, self)
        # [N, 128, 3]
        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)

        # get the sdf_bg
        with torch.no_grad():
            sdf_bg = self.implicit_network.get_specific_sdf_vals(points.reshape(-1, 3), 1)
        sdf_bg = sdf_bg.reshape(cam_loc.shape[0], -1)
        # Create mask for valid points where the first point is not occupied
        mask_0_not_occupied = sdf_bg[:, 0] > 0 # [N, N_samples], true value means in the room and false means outside the room

        # Calculate if sign change occurred and concat 1 (no sign change) in
        # last dimension
        sign_matrix = torch.cat([torch.sign(sdf_bg[:, :-1] * sdf_bg[:, 1:]),
                                 torch.ones(sdf_bg.shape[0], 1).to(sdf_bg.device)],
                                dim=-1)
        cost_matrix = sign_matrix * torch.arange(
            sdf_bg.shape[1], 0, -1).float().to(sdf_bg.device)

        # Get first sign change and mask for values where a.) a sign changed
        # occurred and b.) no a neg to pos sign change occurred (meaning from
        # inside surface to outside)
        values, indices = torch.min(cost_matrix, -1)
        mask_sign_change = values < 0

        # NOTE: We do not to use secant method add more sampling steps
        # Define mask where a valid depth value is found
        n, n_steps = sdf_bg.shape
        mask_pos_to_neg = sdf_bg[torch.arange(n), indices] > 0

        mask = mask_sign_change & mask_0_not_occupied & mask_pos_to_neg

        # Get depth values and function values for the interval
        # to which we want to apply the Secant method
        # [N_rays, 1]
        d_low = z_vals.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)][mask]
        f_low = sdf_bg.view(n, n_steps, 1)[torch.arange(n), indices.view(n)][mask]
        indices = torch.clamp(indices + 1, max=n_steps-1)
        d_high = z_vals.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)][mask]
        f_high = sdf_bg.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)][mask]

        ray0_masked = cam_loc[mask]
        ray_direction_masked = ray_dirs[mask]
        # Apply surface depth refinement step (e.g. Secant method)
        d_pred = self.secant(
            f_low, f_high, d_low, d_high, 8, ray0_masked,
            ray_direction_masked, tau=0., idx=1)

        # for sanity
        d_pred_out = torch.ones(n, 1).to(cam_loc.device)
        d_pred_out[mask] = d_pred
        d_pred_out[mask == 0] = far[mask == 0]
        d_pred_out[mask_0_not_occupied == 0] = far[mask_0_not_occupied == 0]

        d_pred_out = d_pred_out.reshape(-1, 1)
        points_floor = cam_loc + d_pred_out * ray_dirs
        return points_floor, mask

    def secant(self, f_low, f_high, d_low, d_high, n_secant_steps,
                          ray0_masked, ray_direction_masked, tau, idx=0):
        ''' Runs the secant method for interval [d_low, d_high].
        Args:
            d_low (tensor): start values for the interval, actually for sdf this should be > 0
            d_high (tensor): end values for the interval, actually for sdf this should be < 0
            n_secant_steps (int): number of steps
            ray0_masked (tensor): masked ray start points
            ray_direction_masked (tensor): masked ray direction vectors
            model (nn.Module): model model to evaluate point occupancies
            c (tensor): latent conditioned code c
            tau (float): threshold value in logits
        '''
        # this guarantee a valid depth no matter f_low > 0 or f_low < 0
        d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
        for i in range(n_secant_steps):
            p_mid = ray0_masked + d_pred * ray_direction_masked
            with torch.no_grad():
                f_mid = self.implicit_network.get_specific_sdf_vals(p_mid, idx=idx)[...,0] - tau
                f_mid = f_mid.unsqueeze(-1)
            
            # ind_low = f_mid < 0
            # ind_low = ind_low
            ind_low = torch.sign(f_mid * f_low)
            ind_low[ind_low <= 0] = 0
            ind_low = ind_low.long()

            if ind_low.sum() > 0:
                d_low[ind_low == 1] = d_pred[ind_low == 1]
                f_low[ind_low == 1] = f_mid[ind_low == 1]
            if (ind_low == 0).sum() > 0:
                d_high[ind_low == 0] = d_pred[ind_low == 0]
                f_high[ind_low == 0] = f_mid[ind_low == 0]

            d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
        return d_pred
    
    def get_bg_surface(self, val):
        '''
        ray_dirs: [N, 3]
        According to indoor scene initialization, sdf should be first > 0 then < 0
        val: [N, N_samples]
        '''
        # Create mask for valid points where the first point is not occupied
        mask_0_not_occupied = val > 0 # [N, N_samples], true value means in the room and false means outside the room

        # Calculate if sign change occurred and concat 1 (no sign change) in
        # last dimension
        sign_matrix = torch.cat([torch.sign(val[:, :-1] * val[:, 1:]),
                                 torch.ones(val.shape[0], 1).to(val.device)],
                                dim=-1)
        cost_matrix = sign_matrix * torch.arange(
            val.shape[1], 0, -1).float().to(val.device)

        # Get first sign change and mask for values where a.) a sign changed
        # occurred and b.) no a neg to pos sign change occurred (meaning from
        # inside surface to outside)
        values, indices = torch.min(cost_matrix, -1)
        mask_sign_change = values < 0
        indices[~mask_sign_change] = val.shape[1] - 1
        return indices