"""
    From model_v2.py
        encapsule network into MaskNet and PoseNet
        No delta, mask use 0/1 values
"""

import os
import sys
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.models import resnet18
from cd.chamfer import chamfer_distance
from quaternion import qrot, qmul
from utils import load_pts, get_surface_reweighting_batch, transform_pc_batch
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../utils', 'sampling'))
from sampling import furthest_point_sample
import torch_scatter


class UNet(nn.Module):

    def __init__(
        self,
        cond_feat_dim,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=5,
        padding=False,
        batch_norm=False,
        up_mode='upconv',
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')

        self.padding = padding
        self.depth = depth
        prev_channels = in_channels

        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            if i == depth - 2:
                self.up_path.append(
                    UNetUpBlock(prev_channels, cond_feat_dim, 2 ** (wf + i), up_mode, padding, batch_norm)
                )
            else:
                self.up_path.append(
                    UNetUpBlock(prev_channels, 0, 2 ** (wf + i), up_mode, padding, batch_norm)
                )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x, feat):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        feat_size = x.shape[2]
        batch_size = feat.shape[0]
        feat = feat.reshape(batch_size, -1, 1, 1)
        feat = feat.repeat(1, 1, feat_size, feat_size)
        x = torch.cat([x, feat], dim=1)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        x = self.last(x)
        return x


class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):

    def __init__(self, in_size, add_in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()

        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size + add_in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        batch_size = up.shape[0]
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


class PointNet(nn.Module):
    def __init__(self, emb_dim):
        super(PointNet, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, emb_dim, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(emb_dim)

        self.mlp1 = nn.Linear(emb_dim, emb_dim)
        self.bn6 = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        batch_size = x.shape[0]

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))

        x = x.max(dim=-1)[0]

        x = torch.relu(self.bn6(self.mlp1(x)))
        return x


class MaskNet(nn.Module):

    def __init__(self, conf, sem_cnt ):
        super(MaskNet, self).__init__()

        self.conf = conf
        # self.box_dim = 3
        self.img_size = conf.img_size
        self.sem_dim = sem_cnt
        self.ins_dim = conf.ins_dim
        self.pointnet = PointNet(conf.pointnet_emd_dim)

        self.part_dim_len = conf.pointnet_emd_dim + conf.ins_dim  #+ self.box_dim
        if conf.use_semantics:
            self.part_dim_len += sem_cnt

        ## global part feature network
        self.mlp1 = nn.Linear(self.part_dim_len, 256)
        self.mlp2 = nn.Linear(256, 256)

        ## mask net work
        self.unet = UNet(self.part_dim_len + 256, \
            in_channels=3, n_classes=1, padding=True, batch_norm=True)
        self.softmax = nn.Softmax(dim=0)


    def forward(self, img, pc, sem_feat, ins_feat, part_cnt):
        '''
            input:
            img:            max_parts x 3 x H x W
            pc :            max_parts x N x 3
            sem_feat        max_parts x K
            ins_feat        max_parts x max_similar parts
            similar_cnt     max_parts x 2

            output:         part_cnt+1 x H x W

        '''
        batch_size = img.shape[0]
        pc_feat = self.pointnet(pc)

        if self.conf.use_semantics:
            part_feat = torch.cat([pc_feat, sem_feat, ins_feat], dim=1)
        else:
            part_feat = torch.cat([pc_feat, ins_feat], dim=1)

        part_feat_3d = torch.relu(self.mlp1(part_feat))
        
        pc_global_feat = []
        t = 0

        for cnt in part_cnt:
            cur_global_feat = part_feat_3d[t:t+cnt].max(dim=0, keepdim=True)[0]
            cur_global_feat = torch.relu(self.mlp2(cur_global_feat)).repeat(cnt, 1)
            pc_global_feat.append(cur_global_feat)
            t += cnt

        pc_global_feat = torch.cat(pc_global_feat, dim=0)

        cond_feat = torch.cat( (part_feat, pc_global_feat), dim=1)

        masks = self.unet(img, cond_feat).squeeze(1)
        
        t = 0 ; i = 0

        pred_masks = []
        for cnt in part_cnt:
            background = Variable(torch.zeros(1,  self.img_size, self.img_size, dtype = torch.float), requires_grad=True).to(self.conf.device)
            pred_mask = self.softmax( torch.cat( (masks[t:t+cnt], background), dim=0))
            pred_masks.append( pred_mask[:-1] )
            t += cnt

        return pred_masks, cond_feat

class PoseDecoder(nn.Module):

    def __init__(self, input_feat_len):
        super(PoseDecoder, self).__init__()

        self.mlp = nn.Linear(input_feat_len, 256)
        
        self.center = nn.Linear(256, 3)
        
        self.quat = nn.Linear(256, 4)
        self.quat.bias.data.zero_()

    def forward(self, feat):
        feat = torch.relu(self.mlp(feat))
        
        center = self.center(feat)
        
        quat_bias = feat.new_tensor([[1.0, 0.0, 0.0, 0.0]])
        quat = self.quat(feat).add(quat_bias)
        quat = quat / (1e-12 + quat.pow(2).sum(dim=1).unsqueeze(dim=1).sqrt())

        return center, quat


class PointNet2(nn.Module):
    def __init__(self, emb_dim):
        super(PointNet2, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
                
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.mlp1 = nn.Linear(1024, emb_dim)
        self.bn6 = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        batch_size = x.shape[0]

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))

        x = x.max(dim=-1)[0]

        x = torch.relu(self.bn6(self.mlp1(x)))
        return x


class RefineNet(nn.Module):

    def __init__(self, conf):
        super(RefineNet, self).__init__()

        self.mlp1 = nn.Linear(256*4+3+4, 256)

        self.pointnet = PointNet2(256)
        
        self.pose_decoder = PoseDecoder(256*3)
        
        # graph-conv
        self.node_edge_op = torch.nn.ModuleList()
        self.num_iteration = 2
        for i in range(self.num_iteration):
            self.node_edge_op.append(nn.Linear(256*2+1, 256))


    def forward(self, part_pcs, part_feat_old, part_cnt, equiv_edge_indices):
        after_part_feat_3d = self.pointnet(part_pcs)
        
        part_feat_8d = torch.cat([part_feat_old, after_part_feat_3d], dim=1)
        part_feat_8d = torch.relu(self.mlp1(part_feat_8d))
        
        # perform graph-conv
        t = 0; i = 0; output_part_feat_8d = [];
        for cnt in part_cnt:
            child_feats = part_feat_8d[t:t+cnt]
            iter_feats = [child_feats]
            
            cur_equiv_edge_indices = equiv_edge_indices[i]
            cur_equiv_edge_feats = cur_equiv_edge_indices.new_zeros(cur_equiv_edge_indices.shape[0], 1)

            # detect adj
            topk = min(5, cnt-1)
            with torch.no_grad():
                cur_part_pcs = part_pcs[t:t+cnt]
                A = cur_part_pcs.unsqueeze(0).repeat(cnt, 1, 1, 1).view(cnt*cnt, -1, 3)
                B = cur_part_pcs.unsqueeze(1).repeat(1, cnt, 1, 1).view(cnt*cnt, -1, 3)
                dist1, dist2 = chamfer_distance(A, B, transpose=False)
                dist = dist1.min(dim=1)[0].view(cnt, cnt)
                cur_adj_edge_indices = []
                for j in range(cnt):
                    for k in dist[j].argsort()[1:topk+1]:
                        cur_adj_edge_indices.append([j, k.item()])
                cur_adj_edge_indices = torch.Tensor(cur_adj_edge_indices).long().to(cur_equiv_edge_indices.device)
                cur_adj_edge_feats = cur_adj_edge_indices.new_ones(cur_adj_edge_indices.shape[0], 1)

            cur_edge_indices = torch.cat([cur_equiv_edge_indices, cur_adj_edge_indices], dim=0)
            cur_edge_feats = torch.cat([cur_equiv_edge_feats, cur_adj_edge_feats], dim=0).float()

            for j in range(self.num_iteration):
                node_edge_feats = torch.cat([child_feats[cur_edge_indices[:, 0], :], child_feats[cur_edge_indices[:, 1], :], cur_edge_feats], dim=1)
                node_edge_feats = torch.relu(self.node_edge_op[j](node_edge_feats))
                new_child_feats = child_feats.new_zeros(cnt, 256)
                new_child_feats = torch_scatter.scatter_mean(node_edge_feats, cur_edge_indices[:, 0], dim=0, out=new_child_feats)
                child_feats = new_child_feats
                iter_feats.append(child_feats)

            all_iter_feat = torch.cat(iter_feats, dim=1)
            output_part_feat_8d.append(all_iter_feat)
            t += cnt; i += 1;

        feat = torch.cat(output_part_feat_8d, dim=0)
               
        center, quat = self.pose_decoder(feat)
        
        return center, quat


class PoseNet(nn.Module):

    def __init__(self, conf, sem_cnt):
        super(PoseNet, self).__init__()

        # self.box_dim = 3
        self.global_dim_len = conf.pointnet_emd_dim + conf.ins_dim + 2*conf.resnet_feat_dim + 256 # + self.box_dim 
        if conf.use_semantics:
            self.global_dim_len += sem_cnt

        self.resnet = resnet18(pretrained=conf.pretrain_resnet)
        self.resnet.fc = nn.Linear(in_features=512, out_features=conf.resnet_feat_dim, bias=True)
        self.resnet_final_bn = nn.BatchNorm1d(conf.resnet_feat_dim)

        self.mask_resnet = resnet18(pretrained=False)
        self.mask_resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=True)
        self.mask_resnet.fc = nn.Linear(in_features=512, out_features=conf.resnet_feat_dim, bias=True)
        self.mask_resnet_final_bn = nn.BatchNorm1d(conf.resnet_feat_dim)

        self.pose_decoder = PoseDecoder(256*3)

        ## 2d and 3d global feature network 
        self.mlp3 = nn.Linear(self.global_dim_len, 256)

        # graph-conv
        self.node_edge_op = torch.nn.ModuleList()
        self.num_iteration = 2
        for i in range(self.num_iteration):
            self.node_edge_op.append(nn.Linear(256*2, 256))

        ## refine final poses
        self.refine_net = RefineNet(conf)

    def forward(self, img, pc, pred_masks, part_feat, part_cnt, equiv_edge_indices):
        # get image feature
        img = img.clone()
        img[:, 0] = (img[:, 0] - 0.485) / 0.229
        img[:, 1] = (img[:, 1] - 0.456) / 0.224
        img[:, 2] = (img[:, 2] - 0.406) / 0.225
        img_feat = torch.relu(self.resnet_final_bn(self.resnet(img)))

        # get mask feature
        all_masks = (torch.cat(pred_masks, dim=0).unsqueeze(dim=1) > 0.5).float()
        mask_feat = torch.relu(self.mask_resnet_final_bn(self.mask_resnet(all_masks)))

        # get global feature per shape
        part_feat_5d = torch.cat((img_feat, mask_feat, part_feat), dim=1)
        part_feat_5d = torch.relu(self.mlp3(part_feat_5d))
        
        # perform graph-conv
        t = 0; i = 0; output_part_feat_5d = [];
        for cnt in part_cnt:
            child_feats = part_feat_5d[t:t+cnt]
            iter_feats = [child_feats]
            cur_equiv_edge_indices = equiv_edge_indices[i]
            for j in range(self.num_iteration):
                if cur_equiv_edge_indices.shape[0] > 0:
                    node_edge_feats = torch.cat([child_feats[cur_equiv_edge_indices[:, 0], :], child_feats[cur_equiv_edge_indices[:, 1], :]], dim=1)
                    node_edge_feats = torch.relu(self.node_edge_op[j](node_edge_feats))
                    new_child_feats = child_feats.new_zeros(cnt, 256)
                    new_child_feats = torch_scatter.scatter_mean(node_edge_feats, cur_equiv_edge_indices[:, 0], dim=0, out=new_child_feats)
                    child_feats = new_child_feats
                iter_feats.append(child_feats)
            all_iter_feat = torch.cat(iter_feats, dim=1)
            output_part_feat_5d.append(all_iter_feat)
            t += cnt; i += 1;

        output_part_feat_5d = torch.cat(output_part_feat_5d, dim=0)
        
        # decode pose
        center, quat = self.pose_decoder(output_part_feat_5d)

        # refine final poses
        num_point = pc.shape[1]

        with torch.no_grad():
            pc2 = qrot(quat.unsqueeze(1).repeat(1, num_point, 1), pc) + center.unsqueeze(1).repeat(1, num_point, 1)
        
        feat = torch.cat([output_part_feat_5d, center, quat], dim=1)
        delta_center, delta_quat = self.refine_net(pc2, feat, part_cnt, equiv_edge_indices)

        center2 = center + delta_center
        quat2 = qmul(delta_quat, quat)

        return center, quat, center2, quat2


class Network(nn.Module):

    def __init__(self, conf, sem_cnt ):
        super(Network, self).__init__()
        self.conf = conf

        self.mask_net = MaskNet(conf, sem_cnt)
        self.pose_net = PoseNet(conf, sem_cnt)
        
        self.unit_cube = torch.from_numpy(load_pts('../utils/cube.pts')).to(conf.device)
        self.unit_anchor = torch.from_numpy(load_pts('../utils/anchor.pts')).to(conf.device)

    def forward(self, img, pc, sem_feat, ins_feat, part_cnt, equiv_edge_indices):
        '''
            input:
            img:            max_parts x 3 x H x W
            pc :            max_parts x N x 3
            sem_feat        max_parts x K
            ins_feat        max_parts x max_similar parts
            similar_cnt     max_parts x 2

            output:         part_cnt+1 x H x W

        '''
        batch_size = img.shape[0]
        num_point = pc.shape[1]

        with torch.no_grad():
            pred_masks, part_feat = self.mask_net(img, pc, sem_feat, ins_feat, part_cnt)        
        
        center, quat, center2, quat2 = self.pose_net(img, pc, pred_masks, part_feat, part_cnt, equiv_edge_indices)
        
        return pred_masks, center, quat, center2, quat2

    def get_mask_loss(self, mask1, mask2):
        
        batch_size = mask1.shape[0]

        inter = (mask1 * mask2).mean(dim=[1, 2])
        union = mask1.mean(dim=[1, 2]) + mask2.mean(dim=[1, 2]) - inter
        loss_per_data = - inter / (union + 1e-12)
        return loss_per_data

    def get_center_loss(self, center1, center2):
        loss_per_data = (center1 - center2).pow(2).sum(dim=1)
        return loss_per_data

    def get_quat_loss(self, pts, quat1, quat2):
        num_point = pts.shape[1] 

        pts1 = qrot(quat1.unsqueeze(1).repeat(1, num_point, 1), pts)
        pts2 = qrot(quat2.unsqueeze(1).repeat(1, num_point, 1), pts)

        dist1, dist2 = chamfer_distance(pts1, pts2, transpose=False)
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
        return loss_per_data

    def get_l2_rotation_loss(self, pts, quat1, quat2):

        num_point = pts.shape[1] 

        pts1 = qrot(quat1.unsqueeze(1).repeat(1, num_point, 1), pts)
        pts2 = qrot(quat2.unsqueeze(1).repeat(1, num_point, 1), pts)

        loss_per_data = (pts1 - pts2).pow(2).sum(dim=2).mean(dim=1)

        return loss_per_data

    def get_l2_joint_loss(self, pts, quat1, quat2, center1, center2):
        num_point = pts.shape[1]

        pts1 = qrot(quat1.unsqueeze(1).repeat(1, num_point, 1), pts) + center1.unsqueeze(1).repeat(1, num_point, 1)
        pts2 = qrot(quat2.unsqueeze(1).repeat(1, num_point, 1), pts) + center2.unsqueeze(1).repeat(1, num_point, 1)

        loss_per_data = (pts1 - pts2).pow(2).sum(dim=2).mean(dim=1)

        return loss_per_data



    def get_joint_pose_loss(self, pts, quat1, quat2, center1, center2):
        num_point = pts.shape[1]

        pts1 = qrot(quat1.unsqueeze(1).repeat(1, num_point, 1), pts) + center1.unsqueeze(1).repeat(1, num_point, 1)
        pts2 = qrot(quat2.unsqueeze(1).repeat(1, num_point, 1), pts) + center2.unsqueeze(1).repeat(1, num_point, 1)

        dist1, dist2 = chamfer_distance(pts1, pts2, transpose=False)
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)        

        return loss_per_data

    def get_box_loss(self, box_size, quat1, quat2, center1, center2):
        box1 = torch.cat([center1, box_size, quat1], dim = -1)
        box2 = torch.cat([center2, box_size, quat2], dim = -1)

        box1_pc = transform_pc_batch(self.unit_cube, box1)
        box2_pc = transform_pc_batch(self.unit_cube, box2)

        with torch.no_grad():
            box1_reweight = get_surface_reweighting_batch(box1[:, 3:6], self.unit_cube.size(0))
            box2_reweight = get_surface_reweighting_batch(box2[:, 3:6], self.unit_cube.size(0))
        
        d1, d2 = chamfer_distance(box1_pc, box2_pc, transpose=False)
        loss_per_data =  (d1 * box1_reweight).sum(dim=1) / (box1_reweight.sum(dim=1) + 1e-12) + \
                (d2 * box2_reweight).sum(dim=1) / (box2_reweight.sum(dim=1) + 1e-12)
        return loss_per_data

    def get_anchor_loss(self, box_size, quat1, quat2, center1, center2):
        box1 = torch.cat([center1, box_size, quat1], dim = -1)
        box2 = torch.cat([center2, box_size, quat2], dim = -1)

        anchor1_pc = transform_pc_batch(self.unit_anchor, box1, anchor=True)
        anchor2_pc = transform_pc_batch(self.unit_anchor, box2, anchor=True)

        d1, d2 = chamfer_distance(anchor1_pc, anchor2_pc, transpose=False)
        loss_per_data = (d1.mean(dim=1) + d2.mean(dim=1)) / 2
        return loss_per_data
    
    def get_adj_loss(self, pts, quat, center, adjs):
        num_point = pts.shape[1]

        part_pcs = qrot(quat.unsqueeze(1).repeat(1, num_point, 1), pts) + center.unsqueeze(1).repeat(1, num_point, 1)
        
        loss = []
        for cur_shape_adj in adjs:
            cur_shape_loss = []
            for adj in cur_shape_adj:
                idx1, idx2 = adj
                dist1, dist2 = chamfer_distance(part_pcs[idx1].unsqueeze(0), part_pcs[idx2].unsqueeze(0), transpose=False)
                cur_loss = torch.min(dist1, dim=1)[0] + torch.min(dist2, dim=1)[0]
                cur_shape_loss.append(cur_loss)
            loss.append(torch.stack(cur_shape_loss).mean())
        return loss

    def get_shape_chamfer_loss(self, pts, quat1, quat2, center1, center2, part_cnt):
        num_point = pts.shape[1]
        part_pcs1 = qrot(quat1.unsqueeze(1).repeat(1, num_point, 1), pts) + center1.unsqueeze(1).repeat(1, num_point, 1)
        part_pcs2 = qrot(quat2.unsqueeze(1).repeat(1, num_point, 1), pts) + center2.unsqueeze(1).repeat(1, num_point, 1)
        t = 0; shape_pcs1 = []; shape_pcs2 = []
        for cnt in part_cnt:
            cur_shape_pc1 = part_pcs1[t:t+cnt].view(1, -1, 3)
            cur_shape_pc2 = part_pcs2[t:t+cnt].view(1, -1, 3)
            with torch.no_grad():
                idx1 = furthest_point_sample(cur_shape_pc1, 2048).long()[0]
                idx2 = furthest_point_sample(cur_shape_pc2, 2048).long()[0]
            shape_pcs1.append(cur_shape_pc1[:, idx1])
            shape_pcs2.append(cur_shape_pc2[:, idx2])
            t += cnt
        shape_pcs1 = torch.cat(shape_pcs1, dim=0) # numshapes x 2048 x 3
        shape_pcs2 = torch.cat(shape_pcs2, dim=0) # numshapes x 2048 x 3
        
        dist1, dist2 = chamfer_distance(shape_pcs1, shape_pcs2, transpose=False)
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)        

        return loss_per_data



    def linear_assignment(self, mask1, mask2, similar_cnt, pts, centers1, quats1, centers2, quats2):
        '''
            mask1, mask 2:
                # part_cnt x 224 x 224 
            similar cnt
                # shape: max_mask  x  2
                # first index is the index of parts without similar parts 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, .... 
                # second index is the number of similar part count:       1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, .... 

        '''

        bids = []; ids1 = []; ids2 = []; inds1 = []; inds2 = []; 
        img_size = mask1.shape[-1]
        t = 0

        num_point = pts.shape[1]
        max_num_part = centers1.shape[1]

        # part_cnt = [item for sublist in part_cnt for item in sublist]

        with torch.no_grad():
            while t < similar_cnt.shape[0]:

                cnt = similar_cnt[t].item()
                bids = [t] * cnt
                cur_mask1 = mask1[t:t+cnt].unsqueeze(1).repeat(1, cnt, 1, 1).view(-1, img_size, img_size)
                cur_mask2 = mask2[t:t+cnt].unsqueeze(0).repeat(cnt, 1, 1, 1).view(-1, img_size, img_size)
                

                dist_mat_mask = self.get_mask_loss(cur_mask1, cur_mask2).view(cnt, cnt)

                dist_mat_mask = torch.clamp(dist_mat_mask,  max=-0.1)

                cur_pts = pts[t:t+cnt]
                
                cur_quats1 = quats1[t:t+cnt].unsqueeze(1).repeat(1, num_point, 1)
                cur_centers1 = centers1[t:t+cnt].unsqueeze(1).repeat(1, num_point, 1)
                cur_pts1 = qrot(cur_quats1, cur_pts) + cur_centers1
 
                cur_quats2 = quats2[t:t+cnt].unsqueeze(1).repeat(1, num_point, 1)
                cur_centers2 = centers2[t:t+cnt].unsqueeze(1).repeat(1, num_point, 1)
                cur_pts2 = qrot(cur_quats2, cur_pts) + cur_centers2

                cur_pts1 = cur_pts1.unsqueeze(1).repeat(1, cnt, 1, 1).view(-1, num_point, 3)
                cur_pts2 = cur_pts2.unsqueeze(0).repeat(cnt, 1, 1, 1).view(-1, num_point, 3)
                
                dist1, dist2 = chamfer_distance(cur_pts1, cur_pts2, transpose=False)
                dist_mat_pts = (dist1.mean(1) + dist2.mean(1)).view(cnt, cnt)

                dist_mat_pts = torch.clamp(dist_mat_pts, max=1) * 0.1

                dist_mat = torch.add(dist_mat_mask, dist_mat_pts)

                t += cnt
                rind, cind = linear_sum_assignment(dist_mat.cpu().numpy())
                
                ids1 = list(rind)
                ids2 = list(cind)
                inds1 += [bids[i] + ids1[i] for i in range(len(ids1))]
                inds2 += [bids[i] + ids2[i] for i in range(len(ids2))]
        return inds1, inds2
  
  
    def linear_assignment_old(self, mask1, mask2, similar_cnt):
        '''
            mask1, mask 2:
                # part_cnt x 224 x 224 
            similar cnt
                # shape: max_mask  x  2
                # first index is the index of parts without similar parts 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, .... 
                # second index is the number of similar part count:       1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, .... 

        '''

        bids = []; ids1 = []; ids2 = []; inds1 = []; inds2 = []; 
        img_size = mask1.shape[-1]
        t = 0
        with torch.no_grad():
            while t < similar_cnt.shape[0]:

                cnt = similar_cnt[t].item()
                bids = [t] * cnt
                cur_mask1 = mask1[t:t+cnt].unsqueeze(1).repeat(1, cnt, 1, 1).view(-1, img_size, img_size)
                cur_mask2 = mask2[t:t+cnt].unsqueeze(0).repeat(cnt, 1, 1, 1).view(-1, img_size, img_size)
                dist_mat = self.get_mask_loss(cur_mask1, cur_mask2)
                dist_mat = dist_mat.view(cnt, cnt)

                t += cnt
                rind, cind = linear_sum_assignment(dist_mat.cpu().numpy())
                
                ids1 = list(rind)
                ids2 = list(cind)
                inds1 += [bids[i] + ids1[i] for i in range(len(ids1))]
                inds2 += [bids[i] + ids2[i] for i in range(len(ids2))]
        return inds1, inds2
  
