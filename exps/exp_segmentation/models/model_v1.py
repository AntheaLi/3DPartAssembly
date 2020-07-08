"""
    UNet + part PointNet feature + sematnic one hot + instance one hot concatenated in bottleneck
    Input: 
        a shape image:                  3 x 224 x 224
        a normalized part point-cloud:  Part Cnt x 1000 x 3
        sem one-hot:                    Part Cnt x K (num of semantics, e.g. Chair 57)
        instance one hot                Part Cnt x M (max number of similar parts, )
    Output: 
        part mask:                      Part Cnt x 224 x 224
    Loss:
        neg Soft-IoU
"""

import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

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


class Network(nn.Module):

    def __init__(self, conf, partleaf_cnt ):
        super(Network, self).__init__()
        self.conf = conf
        self.img_size = conf.img_size
        self.sem_dim = partleaf_cnt
        self.ins_dim = conf.ins_dim
        self.pointnet = PointNet(conf.pointnet_emd_dim)
        self.dim_len = conf.pointnet_emd_dim + conf.ins_dim
        if conf.use_semantics:
            self.dim_len = conf.pointnet_emd_dim + self.sem_dim + conf.ins_dim
        self.unet = UNet(self.dim_len, \
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
        pc_feat = self.pointnet(pc)

        if self.conf.use_semantics:
            cond_feat = torch.cat([pc_feat, sem_feat, ins_feat], dim=1)
        else:
            cond_feat = torch.cat([pc_feat, ins_feat], dim=1)

        masks = self.unet(img, cond_feat).squeeze(1)
        
        t = 0
        all_masks = ()
        for cnt in part_cnt:
            background = torch.zeros(1,  self.img_size, self.img_size).to(self.conf.device)
            cur_mask = self.softmax( torch.cat( (masks[t:t+cnt], background), 0))
            all_masks = all_masks + (cur_mask,)
            t += cnt

        return all_masks

    def get_mask_loss(self, mask1, mask2):
        
        batch_size = mask1.shape[0]

        inter = (mask1 * mask2).mean(dim=[1, 2])
        union = mask1.mean(dim=[1, 2]) + mask2.mean(dim=[1, 2]) - inter
        loss_per_data = - inter / (union + 1e-12)
        return loss_per_data

    def linear_assignment(self, mask1, mask2, similar_cnt):
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
  
