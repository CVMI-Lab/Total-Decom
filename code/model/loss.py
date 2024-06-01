import torch
from torch import nn
import utils.general as utils
import math
import torch.nn.functional as F

# copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask, scale=None, shift=None):

        if scale == None and shift == None:
            scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total, scale, shift

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
# end copy

class SensorDepthLoss(nn.Module):
    """Sensor Depth loss"""

    def __init__(self, truncation: float):
        super(SensorDepthLoss, self).__init__()
        self.truncation = truncation  #  0.05 * 0.3 5cm scaled

    def forward(self, depth_pred, depth_gt, z_vals, pred_sdf):
        """take the mim

        Args:
            batch (Dict): inputs
            outputs (Dict): outputs data from surface model

        Returns:
            l1_loss: l1 loss
            freespace_loss: free space loss
            sdf_loss: sdf loss
        """
        # TODO: Check the dimension here
        # depth_pred = outputs["depth"]
        # depth_gt = batch["sensor_depth"].to(depth_pred.device)[..., None]
        valid_gt_mask = depth_gt > 0.0

        l1_loss = torch.sum(valid_gt_mask * torch.abs(depth_gt - depth_pred)) / (valid_gt_mask.sum() + 1e-6)

        # free space loss and sdf loss
        # ray_samples = outputs["ray_samples"]
        # filed_outputs = outputs["field_outputs"]
        # pred_sdf = filed_outputs[FieldHeadNames.SDF][..., 0]
        # directions_norm = outputs["directions_norm"]

        # z_vals = ray_samples.frustums.starts[..., 0] / directions_norm

        truncation = self.truncation
        front_mask = valid_gt_mask & (z_vals < (depth_gt - truncation))
        back_mask = valid_gt_mask & (z_vals > (depth_gt + truncation))
        sdf_mask = valid_gt_mask & (~front_mask) & (~back_mask)

        num_fs_samples = front_mask.sum()
        num_sdf_samples = sdf_mask.sum()
        num_samples = num_fs_samples + num_sdf_samples + 1e-6
        fs_weight = 1.0 - num_fs_samples / num_samples
        sdf_weight = 1.0 - num_sdf_samples / num_samples

        free_space_loss = torch.mean((F.relu(truncation - pred_sdf) * front_mask) ** 2) * fs_weight

        sdf_loss = torch.mean(((z_vals + pred_sdf) - depth_gt) ** 2 * sdf_mask) * sdf_weight

        return l1_loss, free_space_loss, sdf_loss
    
    
class MonoSDFLoss(nn.Module):
    def __init__(self, rgb_loss, 
                 eikonal_weight, 
                 smooth_weight = 0.005,
                 depth_weight = 0.1,
                 normal_l1_weight = 0.05,
                 normal_cos_weight = 0.05,
                 manhattan_weight = 0.05,
                 sem_weight = 0.1,
                 sdf_weight = 0.1,
                 feature_loss_weight = 0.1,
                 layout_weight = 0.04,
                 end_step = -1):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.smooth_weight = smooth_weight
        self.depth_weight = depth_weight
        self.normal_l1_weight = normal_l1_weight
        self.normal_cos_weight = normal_cos_weight
        self.sem_weight = sem_weight
        self.manhattan_weight = manhattan_weight
        self.sdf_weight = sdf_weight
        self.layout_weight = layout_weight
        self.rgb_loss = utils.get_class(rgb_loss)(reduction='mean')
        self.theta = nn.Parameter(torch.Tensor([0.]).to(torch.device('cuda')), requires_grad=True)
        
        # NOTE: Here the truncation of sensor depth loss is hard coded.
        # self.depth_loss = SensorDepthLoss(truncation=0.05 / 3.65)
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)
        self.semantic_loss_layout = nn.CrossEntropyLoss(ignore_index=-1)
        self.semantic_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.feature_loss_weight = feature_loss_weight
        
        print(f"using weight for loss RGB_1.0 EK_{self.eikonal_weight} SM_{self.smooth_weight} Depth_{self.depth_weight} NormalL1_{self.normal_l1_weight} NormalCos_{self.normal_cos_weight}")
        
        self.step = 0
        self.end_step = end_step

    def get_rgb_loss(self, rgb_values, rgb_gt):
        rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_smooth_loss(self,model_outputs):
        # smoothness loss as unisurf
        g1 = model_outputs['grad_theta']
        g2 = model_outputs['grad_theta_nei']
        
        normals_1 = g1 / (g1.norm(2, dim=1).unsqueeze(-1) + 1e-5)
        normals_2 = g2 / (g2.norm(2, dim=1).unsqueeze(-1) + 1e-5)
        smooth_loss =  torch.norm(normals_1 - normals_2, dim=-1).mean()
        return smooth_loss
    
    def get_depth_loss(self, depth_pred, depth_gt, mask, scale=None, shift=None):
        # TODO remove hard-coded scaling for depth
        # We found that the LSE method can help we get better reconstruction methods.
        return self.depth_loss(depth_pred.reshape(1, 32, 32), 
                               (depth_gt * 50 + 0.5).reshape(1, 32, 32), 
                               mask.reshape(1, 32, 32), scale, shift)
        
    def get_normal_loss(self, normal_pred, normal_gt):
        normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
        normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
        l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1).mean()
        cos = (1. - torch.sum(normal_pred * normal_gt, dim = -1)).mean()
        return l1, cos
    
    def compute_grad_error(self, x, mask):
        scales = 4
        grad_loss = torch.tensor(0.0).cuda().float()
        for i in range(scales):
            step = pow(2, i)

            mask_step = mask[:, ::step, ::step].cuda()
            x_step = x[:, ::step, ::step]

            M = torch.sum(mask_step[:1], (1, 2))

            diff = torch.mul(mask_step, x_step)

            grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
            mask_x = torch.mul(mask_step[:, :, 1:], mask_step[:, :, :-1])
            grad_x = torch.mul(mask_x, grad_x)

            grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
            mask_y = torch.mul(mask_step[:, 1:, :], mask_step[:, :-1, :])
            grad_y = torch.mul(mask_y, grad_y)

            image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

            divisor = torch.sum(M)

            if divisor == 0:
                scale_loss = torch.tensor(0.0).cuda().float()
            else:
                scale_loss = torch.sum(image_loss) / divisor

            grad_loss += scale_loss

        return grad_loss

    def get_manhattan_loss(self, normal_pred, semantic_score, floor_mask, wall_mask, ceiling_mask):
        # The points on the floor use the vertical normal to compute the manhattan constraint
        if floor_mask.sum() > 0:
            floor_normals = normal_pred[floor_mask]
            floor_loss = (semantic_score[floor_mask][:, 0] * (1 - floor_normals[:, 2])).mean()
        else:
            floor_loss = 0.

        # The points on the wall have multiple directions, we need to select them
        if wall_mask.sum() > 0:
            wall_normals = normal_pred[wall_mask]
            wall_loss_vertical = (semantic_score[wall_mask][:, 1] * wall_normals[..., 2].abs()).mean()
            cos = wall_normals[..., 0] * torch.cos(self.theta) + wall_normals[..., 1] * torch.sin(self.theta)
            wall_loss_horizontal = semantic_score[wall_mask][:, 1] * torch.min(cos.abs(), torch.min((1 - cos).abs(), (1 + cos).abs())) # Eq.9
            wall_loss = wall_loss_vertical + wall_loss_horizontal.mean()
        else:
            wall_loss = 0.

        # Ceiling mask has negative effect on the final results
        # if ceiling_mask.sum() > 0:
        #     ceiling_normals = normal_pred[ceiling_mask]
        #     ceiling_loss = (semantic_score[ceiling_mask][:, 2] * (-1 - ceiling_normals[:, 2])).mean()
        # else:
        #     ceiling_loss = 0.

        # manhattan_loss = wall_loss + floor_loss + ceiling_loss
        manhattan_loss = wall_loss + floor_loss

        return manhattan_loss

    def object_distinct_loss(self, sdf_value, min_sdf):
        _, min_indice = torch.min(sdf_value.squeeze(), dim=1, keepdims=True)
        input = -sdf_value.squeeze() - min_sdf.detach()
        res = torch.relu(input).sum(dim=1, keepdims=True) - torch.relu(torch.gather(input, 1, min_indice))
        loss = res.mean()
        return loss

    def object_opacity_loss(self, predict_opacity, gt_opacity, weight=None):
        # normalize predict_opacity
        # predict_opacity = torch.nn.functional.normalize(predict_opacity, p=1, dim=-1)
        target = torch.nn.functional.one_hot(gt_opacity.squeeze(), num_classes=predict_opacity.shape[1]).float()
        if weight is None:
            loss = F.binary_cross_entropy(predict_opacity.clamp(1e-4, 1-1e-4), target)
        return loss

    def foreground_mask_loss(self, sdf_bg, sdf_fg, bg_mask):
        # return torch.relu(-sdf_fg*bg_mask).sum(dim=1).mean()
        # return torch.relu(-sdf_fg*bg_mask).sum(dim=1).mean() + torch.relu(sdf_bg*bg_mask).sum(dim=1).mean()
        return torch.relu(sdf_bg*bg_mask).sum(dim=1).mean()

    def get_floor_constraint_from_ceiling(self, normals_floor, ceiling_mask):
        normals_floor = normals_floor[ceiling_mask]

        normals_floor_loss = (1 - normals_floor[:, 2]).mean()
        return normals_floor_loss

    def forward(self, model_outputs, ground_truth):
        rgb_gt = ground_truth['rgb'].cuda()
        # feature_gt = ground_truth['feature'].cuda().reshape(-1, 256)
        # monocular depth and normal
        depth_gt = ground_truth['depth'].cuda()
        normal_gt = ground_truth['normal'].cuda()
        layout_gt = ground_truth['layout'].cuda()
        sem_gt = ground_truth['sem'].cuda()
        bg_mask = sem_gt != 3 # (1, 1024, 1)
        # selected_mask = mask_1
        # fg_mask = ~mask_1.bool()
        # feature_pred = model_outputs['feature_values']
        normals_floor = model_outputs['normals_floor']

        # All field
        depth_pred = model_outputs['depth_values']
        normal_pred = model_outputs['normal_map'][None]
        layout_pred = model_outputs['layout_probs']
        sem_pred = model_outputs['sem_probs']
        semantic_score = torch.nn.functional.softmax(sem_pred, dim=-1)
        fg_mask = layout_gt.squeeze(0) == 0

        # Processing the bg field
        normal_pred_bg = model_outputs['normal_map_bg'][None]

        # get the absolute normal map to compute the manhattan constraint
        # the manhattan constaint is only applied to the unobserved floor and wall
        normal_pred_abs = model_outputs['normal_bg_map_abs']
        sem_pred_label = torch.argmax(sem_pred, dim=-1)
        floor_mask = sem_pred_label == 0
        wall_mask = sem_pred_label == 1
        ceiling_mask = sem_pred_label == 2
        # (batch,)
        manhattan_loss = self.get_manhattan_loss(normal_pred_abs, semantic_score, floor_mask, wall_mask, ceiling_mask)

        # NOTE: Here we seperate the semantic loss and layout loss
        # layout_loss = self.semantic_loss_layout(layout_pred.squeeze(), layout_gt.squeeze().long())
        layout_loss = self.object_opacity_loss(model_outputs['object_opacity'], layout_gt.squeeze().long())
        
        # NOTE: TEST HERE
        # opacity_bg_sup = torch.zeros_like(layout_gt)
        # opacity_bg_sup[:, 1] = 1
        # opacity_bg_loss = self.object_opacity_loss(model_outputs['object_opacity_bg'], opacity_bg_sup.squeeze().long())
        
        # Only supervise the background mask and only compute the semantic for the bg field
        semantic_loss = self.semantic_loss(sem_pred.squeeze()[bg_mask.squeeze()], 
                                           sem_gt.squeeze().long()[bg_mask.squeeze()])

        if fg_mask.sum() > 0:
            semantic_loss += (semantic_score[:, -1:] * fg_mask).sum() / (fg_mask.sum() + 1e-6)

        # Compute the semantic loss for the background field
        
        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)

        # feature_loss = ((feature_gt - feature_pred) ** 2).sum(-1).mean()
        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        else:
            eikonal_loss = torch.tensor(0.0).cuda().float()

        # only supervised the foreground normal
        mask = ((model_outputs['sdf'] > 0.).any(dim=-1) & (model_outputs['sdf'] < 0.).any(dim=-1))[None, :, None]
        # combine with GT
        mask = (ground_truth['mask'] > 0.5).cuda() & mask

        # NOTE: Here, we supervised the complex and background depth for reconstruction
        # This is the pseudo depth loss
        depth_loss, scale, shift = self.get_depth_loss(depth_pred, depth_gt, mask)
        # bg_depth_loss, _, _ = self.get_depth_loss(depth_pred_bg.reshape(1, -1, 1) * bg_mask.reshape(1, -1, 1), 
        #                                   (depth_gt * bg_mask).reshape(1, -1, 1), 
        #                                   (mask * bg_mask).reshape(1, -1, 1), 
        #                                   scale=scale, shift=shift)
        # depth_loss += bg_depth_loss
        if model_outputs['bg_surface_mask'].sum() > 0 and self.step > 1000:
            foreground_mask_loss = self.foreground_mask_loss(model_outputs['sdf_bg'], 
                                                            model_outputs['sdf_fg'], 
                                                            model_outputs['bg_surface_mask'])
        else:
            foreground_mask_loss = torch.tensor(0.0).cuda().float()

        normal_l1, normal_cos = self.get_normal_loss(normal_pred * mask, normal_gt)
        
        mask = mask & bg_mask.cuda()
        # normal_l1_bg, normal_cos_bg = self.get_normal_loss(normal_pred_bg * bg_mask.cuda(), normal_gt * bg_mask.cuda())
        # # compute the bg depth loss
        # normal_l1 += normal_l1_bg
        # normal_cos += normal_cos_bg

        sample_sdf_loss = self.object_distinct_loss(model_outputs['sample_sdf'], model_outputs['sample_minsdf'])
        if ceiling_mask.sum() > 0 and normals_floor is not None and self.step > 1000:
            normals_floor_loss = self.get_floor_constraint_from_ceiling(normals_floor, ceiling_mask)
        else:
            normals_floor_loss = torch.tensor(0.0).cuda().float()

        if isinstance(depth_loss, float):
            depth_loss = torch.tensor(0.0).cuda().float()    
        
        smooth_loss = self.get_smooth_loss(model_outputs)
        
        # compute decay weights 
        if self.end_step > 0:
            decay = math.exp(-self.step / self.end_step * 10.)
        else:
            decay = 1.0
            
        self.step += 1

        loss = rgb_loss + \
               self.layout_weight * layout_loss + \
               self.sem_weight * semantic_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.smooth_weight * smooth_loss + \
               decay * self.depth_weight * depth_loss + \
               decay * self.normal_l1_weight * normal_l1 + \
               decay * self.normal_cos_weight * normal_cos + \
               decay * self.manhattan_weight * manhattan_loss + \
               self.sdf_weight * sample_sdf_loss +\
               0.01 * foreground_mask_loss +\
               0.01 * normals_floor_loss
            #    self.feature_loss_weight * feature_loss + \
        
        output = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'smooth_loss': smooth_loss,
            'depth_loss': depth_loss,
            'normal_l1': normal_l1,
            'normal_cos': normal_cos,
            'semantic': semantic_loss,
            'layout': layout_loss,
            # 'feature': feature_loss,
            # 'sample_sdf_loss': sample_sdf_loss
        }

        return output
