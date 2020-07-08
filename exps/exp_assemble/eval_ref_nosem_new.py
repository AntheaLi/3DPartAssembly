'''
    for total chamfer + all five losses
'''
import os
import sys
import shutil
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import torch
import utils
from data import PartNetShapeDataset
from subprocess import call
from progressbar import ProgressBar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import render_using_blender as render_utils
from colors import colors
from cd.chamfer import chamfer_distance
from quaternion import qrot

# test parameters
parser = ArgumentParser()
parser.add_argument('--exp_name', type=str, help='name of the training run')
parser.add_argument('--data_fn', type=str, help='test data fn')
parser.add_argument('--result_suffix', type=str, default='nothing')
parser.add_argument('--model_epoch', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--visu_batch', type=int, default=1)
parser.add_argument('--thresh', type=float, default=0.05)
parser.add_argument('--save', action='store_true', default=False, help='save results? [default: False]')
parser.add_argument('--no_visu', action='store_true', default=False, help='no visu? [default: False]')
parser.add_argument('--log_dir', type=str, default='logs', help='exp logs directory')
parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
parser.add_argument('--view', type=str, default='00', help="nothing to do")
parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if result_dir exists [default: False]')

eval_conf = parser.parse_args()

# load train config
train_conf = torch.load(os.path.join(eval_conf.log_dir, eval_conf.exp_name, 'conf.pth'))
train_conf.max_num_mask = train_conf.max_num_parts
train_conf.ins_dim = train_conf.max_num_similar_parts


# load model
model_def = utils.get_model_module(train_conf.model_version)

# set up device
device = torch.device(eval_conf.device)
print(f'Using device: {device}')

# check if eval results already exist. If so, delete it.
result_dir = os.path.join(eval_conf.log_dir, eval_conf.exp_name, f'{train_conf.model_version}_model_epoch_{eval_conf.model_epoch}-{eval_conf.result_suffix}')
if os.path.exists(result_dir):
    if not eval_conf.overwrite:
        response = input('Eval results directory "%s" already exists, overwrite? (y/n) ' % result_dir)
        if response != 'y':
            sys.exit()
    shutil.rmtree(result_dir)
os.mkdir(result_dir)
print(f'\nTesting under directory: {result_dir}\n')

# output result dir
out_dir = os.path.join(result_dir, 'out')
os.mkdir(out_dir)
test_res_dir = os.path.join(out_dir, 'res')
os.mkdir(test_res_dir)
test_res_matched_dir = os.path.join(out_dir, 'res_matched')
os.mkdir(test_res_matched_dir)

if not eval_conf.no_visu:
    visu_dir = os.path.join(result_dir, 'visu')
    os.mkdir(visu_dir)
    input_img_dir = os.path.join(visu_dir, 'input_img')
    os.mkdir(input_img_dir)
    input_pts_dir = os.path.join(visu_dir, 'input_pts')
    os.mkdir(input_pts_dir)
    gt_mask_dir = os.path.join(visu_dir, 'gt_mask')
    os.mkdir(gt_mask_dir)
    pred_mask_dir = os.path.join(visu_dir, 'pred_mask')
    os.mkdir(pred_mask_dir)
    gt_dof_dir = os.path.join(visu_dir, 'gt_assembly')
    os.mkdir(gt_dof_dir)
    pred_dof_dir = os.path.join(visu_dir, 'pred_assembly')
    os.mkdir(pred_dof_dir)
    pred_dof2_dir = os.path.join(visu_dir, 'pred_assembly2')
    os.mkdir(pred_dof2_dir)
    child_dir = os.path.join(visu_dir, 'child')
    os.mkdir(child_dir)
    info_dir = os.path.join(visu_dir, 'info')
    os.mkdir(info_dir)


# dataset
data_features = ['img', 'pts', 'box_size', 'anchor', 'ins_one_hot' , 'part_id', 'total_parts_cnt' , 'similar_parts_cnt', 'mask' , 'parts_cam_dof' ,'shape_id', 'view_id', 'similar_parts_edge_indices']
dataset = PartNetShapeDataset(train_conf.category, train_conf.data_dir, data_features, data_split="test", \
        max_num_mask = train_conf.max_num_mask, max_num_similar_parts=train_conf.max_num_similar_parts, img_size=train_conf.img_size)
print(dataset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=eval_conf.batch_size, shuffle=False, pin_memory=True, \
        num_workers=0, drop_last=False, collate_fn=utils.collate_feats_with_none, worker_init_fn=utils.worker_init_fn)
 
# create models
network = model_def.Network(train_conf, dataset.get_part_count())
models = [network]
model_names = ['network']

# load pretrained model
print('Loading ckpt from ', os.path.join('logs', eval_conf.exp_name, 'ckpts'), eval_conf.model_epoch)
_ = utils.load_checkpoint(
    models=models, model_names=model_names,
    dirname=os.path.join('logs', eval_conf.exp_name, 'ckpts'),
    epoch=eval_conf.model_epoch,
    strict=True)
print('DONE\n')

# send to device
for m in models:
    m.to(device)

# set models to evaluation mode
for m in models:
    m.eval()

shape_chamfer = []
shape_accu = []
part_correct = 0
total_part_num = 0

# test over all data
with torch.no_grad():
    r = 0
    for batch_id, batch in enumerate(dataloader, 0):
        print('[%d/%d] testing....' % (r, len(dataset)))
        batch_index = 1
        if len(batch) == 0:
            print('batch is none')
            break

        cur_batch_size = len(batch[data_features.index('total_parts_cnt')])
        total_part_cnt = batch[data_features.index('total_parts_cnt')][0]
        input_total_part_cnt = batch[data_features.index('total_parts_cnt')][0]                             # 1
        input_img = batch[data_features.index('img')][0]                                                    # 3 x H x W
        input_img = input_img.repeat(input_total_part_cnt, 1, 1, 1)                            # part_cnt 3 x H x W
        input_pts = batch[data_features.index('pts')][0].squeeze(0)[:input_total_part_cnt]                             # part_cnt x N x 3
        input_ins_one_hot = batch[data_features.index('ins_one_hot')][0].squeeze(0)[:input_total_part_cnt]             # part_cnt x max_similar_parts
        input_similar_part_cnt = batch[data_features.index('similar_parts_cnt')][0].squeeze(0)[:input_total_part_cnt]  # part_cnt x 1    
        input_shape_id = [batch[data_features.index('shape_id')][0]]*input_total_part_cnt
        input_part_id = [batch[data_features.index('part_id')][0]]
        input_view_id = [batch[data_features.index('view_id')][0]]*input_total_part_cnt
        input_box_size = batch[data_features.index('box_size')][0].squeeze(0)[:input_total_part_cnt]                             # part_cnt x N x 3
        input_anchor = batch[data_features.index('anchor')][0].squeeze(0)[:input_total_part_cnt]                             # part_cnt x N x 3
    
        # prepare gt: 
        gt_cam_dof = batch[data_features.index('parts_cam_dof')][0].squeeze(0)[:input_total_part_cnt]
        gt_mask = [batch[data_features.index('mask')][0].squeeze(0)[:input_total_part_cnt].to(train_conf.device)]  
        input_total_part_cnt = [batch[data_features.index('total_parts_cnt')][0]]
        input_similar_parts_edge_indices = [batch[data_features.index('similar_parts_edge_indices')][0].to(train_conf.device)]
        

        while total_part_cnt < 32 and batch_index < cur_batch_size:
            cur_input_cnt = batch[data_features.index('total_parts_cnt')][batch_index]
            total_part_cnt += cur_input_cnt
            if total_part_cnt > 40:
                total_part_cnt -= cur_input_cnt
                batch_index += 1
                continue
            cur_batch_img = batch[data_features.index('img')][batch_index].repeat(cur_input_cnt, 1, 1, 1)
            input_img = torch.cat((input_img, cur_batch_img), dim=0)                
            input_pts = torch.cat((input_pts, batch[data_features.index('pts')][batch_index].squeeze(0)[:cur_input_cnt]), dim=0)                            # B x max_parts x N x 3
            input_box_size = torch.cat((input_box_size, batch[data_features.index('box_size')][batch_index].squeeze(0)[:cur_input_cnt]), dim=0)                            # B x max_parts x N x 3
            input_anchor = torch.cat((input_anchor, batch[data_features.index('anchor')][batch_index].squeeze(0)[:cur_input_cnt]), dim=0)                            # B x max_parts x N x 3
            input_ins_one_hot = torch.cat((input_ins_one_hot, batch[data_features.index('ins_one_hot')][batch_index].squeeze(0)[:cur_input_cnt]), dim=0)    # B x max_parts x max_similar_parts
            input_total_part_cnt.append(batch[data_features.index('total_parts_cnt')][batch_index])                             # 1
            input_similar_part_cnt = torch.cat((input_similar_part_cnt, batch[data_features.index('similar_parts_cnt')][batch_index].squeeze(0)[:cur_input_cnt]), dim=0)  # B x max_parts x 2    
            input_shape_id += [batch[data_features.index('shape_id')][batch_index]] *cur_input_cnt
            input_part_id.append(batch[data_features.index('part_id')][batch_index])
            input_view_id += [batch[data_features.index('view_id')][batch_index]] * cur_input_cnt
            gt_cam_dof = torch.cat( (gt_cam_dof, batch[data_features.index('parts_cam_dof')][batch_index].squeeze(0)[:cur_input_cnt]), dim=0)
            # prepare gt
            gt_mask.append(batch[data_features.index('mask')][batch_index].squeeze(0)[:cur_input_cnt].to(train_conf.device))
            input_similar_parts_edge_indices.append(batch[data_features.index('similar_parts_edge_indices')][batch_index].to(train_conf.device))
            batch_index += 1
        input_img = input_img.to(train_conf.device); input_pts = input_pts.to(train_conf.device); #input_sem_one_hot = input_sem_one_hot.to(train_conf.device); 
        input_similar_part_cnt = input_similar_part_cnt.to(train_conf.device); input_ins_one_hot = input_ins_one_hot.to(train_conf.device)
        gt_cam_dof = gt_cam_dof.to(train_conf.device); input_box_size = input_box_size.to(train_conf.device); input_anchor = input_anchor.to(train_conf.device)
        
        # prepare gt
        gt_center = gt_cam_dof[:, :3]   # B x 3
        gt_quat = gt_cam_dof[:, 3:]     # B x 4
        batch_size = input_img.shape[0]
        num_point = input_pts.shape[1]



        # forward through the network
        pred_masks, pred_center, pred_quat, pred_center2, pred_quat2 = network(input_img - 0.5, input_pts, input_ins_one_hot, input_total_part_cnt, input_similar_parts_edge_indices)
                    
                
        # print(input_part_id)
        if eval_conf.save:
            t = 0; to_save = {}
            for i in range(len(input_total_part_cnt)):
                cnt = input_total_part_cnt[i]
                cur_parts = input_part_id[i]
                
                for j in range(cnt):
                    cur_part_id = cur_parts[j]
                    cur_part_pose = np.hstack([pred_center2[t+j].cpu().numpy(), pred_quat2[t+j].cpu().numpy()])
                    to_save[cur_part_id] = cur_part_pose
                fn = input_shape_id[t]+'-'+input_view_id[t]+'.npy'
                np.save(os.path.join(test_res_dir , fn), to_save)
                print('saved', fn)
                t += cnt

        t = 0
        mask_loss_per_data = []; mask_loss_per_data_all = []
        matched_pred_mask_all = []; matched_gt_mask_all = []; 
        matched_mask_loss_per_data_all = [];
        matched_pred_center_all = []; matched_gt_center_all = []
        matched_pred_center2_all = []; 
        matched_pred_quat_all=[]; matched_gt_quat_all = []
        matched_pred_quat2_all=[];
        matched_ins_onehot_all = []

        for i in range(len(input_total_part_cnt)):
            total_cnt = input_total_part_cnt[i]
            matched_gt_ids, matched_pred_ids = network.linear_assignment(gt_mask[i], pred_masks[i], \
                input_similar_part_cnt[t:t+total_cnt], input_pts[t:t+total_cnt], gt_center[t:t+total_cnt], \
                gt_quat[t:t+total_cnt], pred_center[t:t+total_cnt], pred_quat[t:t+total_cnt])
        

            matched_pred_inds_dict = dict(zip(list(range(len(matched_pred_ids))), matched_pred_ids)) 

            # select the matched data
            matched_pred_mask = pred_masks[i][matched_pred_ids]
            matched_gt_mask = gt_mask[i][matched_gt_ids]

            matched_pred_center = pred_center[t:t+total_cnt][matched_pred_ids]
            matched_pred_center2 = pred_center2[t:t+total_cnt][matched_pred_ids]
            matched_gt_center = gt_center[t:t+total_cnt][matched_gt_ids]

            matched_pred_quat = pred_quat[t:t+total_cnt][matched_pred_ids]
            matched_pred_quat2 = pred_quat2[t:t+total_cnt][matched_pred_ids]
            matched_gt_quat = gt_quat[t:t+total_cnt][matched_gt_ids]

            matched_ins_onehot = input_ins_one_hot[t:t+total_cnt][matched_pred_ids]
            matched_ins_onehot_all.append(matched_ins_onehot)
            
            matched_gt_mask_all.append(matched_gt_mask)
            matched_pred_mask_all.append(matched_pred_mask)

            matched_pred_center_all.append(matched_pred_center)
            matched_pred_center2_all.append(matched_pred_center2)
            matched_gt_center_all.append(matched_gt_center)

            matched_pred_quat_all.append(matched_pred_quat)
            matched_pred_quat2_all.append(matched_pred_quat2)
            matched_gt_quat_all.append(matched_gt_quat)

            # for computing mask soft iou loss
            matched_mask_loss_per_data = network.get_mask_loss(matched_pred_mask, matched_gt_mask)
            matched_mask_loss_per_data_all.append(matched_mask_loss_per_data)
            
            mask_loss_per_data_all.append(matched_mask_loss_per_data)
            mask_loss_per_data.append(matched_mask_loss_per_data.mean())

            t += total_cnt

        matched_ins_onehot_all = torch.cat(matched_ins_onehot_all, dim=0)
        matched_pred_mask_all = torch.cat(matched_pred_mask_all, dim=0)
        matched_gt_mask_all = torch.cat(matched_gt_mask_all, dim=0)
        matched_mask_loss_per_data_all = torch.cat(matched_mask_loss_per_data_all, dim=0)
        matched_pred_quat_all = torch.cat(matched_pred_quat_all, dim=0)
        matched_pred_quat2_all = torch.cat(matched_pred_quat2_all, dim=0)
        matched_gt_quat_all = torch.cat(matched_gt_quat_all, dim=0)
        matched_pred_center_all = torch.cat(matched_pred_center_all, dim=0)
        matched_pred_center2_all = torch.cat(matched_pred_center2_all, dim=0)
        matched_gt_center_all = torch.cat(matched_gt_center_all, dim=0)

        center_loss_per_data = network.get_center_loss(matched_pred_center_all, matched_gt_center_all) + \
                network.get_center_loss(matched_pred_center2_all, matched_gt_center_all)

        quat_loss_per_data = network.get_quat_loss(input_pts, matched_pred_quat_all, matched_gt_quat_all) + \
                network.get_quat_loss(input_pts, matched_pred_quat2_all, matched_gt_quat_all)

        l2_rotation_loss_per_data = network.get_l2_rotation_loss(input_pts, matched_pred_quat_all, matched_gt_quat_all) + \
                network.get_l2_rotation_loss(input_pts, matched_pred_quat2_all, matched_gt_quat_all)

        pose_loss_per_data = network.get_joint_pose_loss(input_pts, matched_pred_quat_all, matched_gt_quat_all, matched_pred_center_all, matched_gt_center_all)
        box_loss_per_data = network.get_box_loss(input_box_size, matched_pred_quat_all, matched_gt_quat_all, matched_pred_center_all, matched_gt_center_all)
        anchor_loss_per_data = network.get_anchor_loss(input_box_size, matched_pred_quat_all, matched_gt_quat_all, matched_pred_center_all, matched_gt_center_all)
        l2_joint_loss_per_data = network.get_l2_joint_loss(input_pts, matched_pred_quat_all, matched_gt_quat_all, matched_pred_center_all, matched_gt_center_all)

        whole_shape_cd_per_shape = network.get_shape_chamfer_loss(input_pts, matched_pred_quat2_all, matched_gt_quat_all, matched_pred_center2_all, matched_gt_center_all, input_total_part_cnt)


        # for each type of loss, compute avg loss per batch
        mask_loss_per_data = torch.stack(mask_loss_per_data)
        center_loss = center_loss_per_data.mean()
        quat_loss = quat_loss_per_data.mean()
        mask_loss = mask_loss_per_data.mean()
        l2_rotation_loss = l2_rotation_loss_per_data.mean()
        shape_chamfer_loss = whole_shape_cd_per_shape.mean()

        # print('shapes', len(mask_loss_per_data), matched_mask_loss_per_data.shape, mask_loss_per_data_all)

        # export results
        for i in range(len(input_total_part_cnt)):
            cur_pred_mask = pred_masks[i].cpu().numpy()
            np.savez(os.path.join(out_dir, '%d.npz' % (t + i)), mask=cur_pred_mask)
            with open(os.path.join(out_dir, '%d.txt' % (t + i)), 'w') as fout:
                fout.write('shape_id: %s, view_id: %s\n' % (\
                        batch[data_features.index('shape_id')][i], \
                        batch[data_features.index('view_id')][i]))
                fout.write('mask_loss: %f\n' % mask_loss_per_data[i].item())
                fout.write('center_loss: %f\n' % center_loss_per_data[i].item())
                fout.write('quat_loss: %f\n' % quat_loss_per_data[i].item())   
                fout.write('pose_loss: %f\n' % pose_loss_per_data[i].item())
                fout.write('box_loss: %f\n' % box_loss_per_data[i].item())
                fout.write('anchor_loss: %f\n' % anchor_loss_per_data[i].item())
                fout.write('l2_rotation_loss: %f\n' % l2_rotation_loss_per_data[i].item())
                fout.write('l2_joint_loss: %f\n' % l2_joint_loss_per_data[i].item())
                fout.write('whole_shape_cd_loss: %f\n' % whole_shape_cd_per_shape[i].item()) 

         
        # print(input_part_id)
        if eval_conf.save:
            t = 0; to_save = {}
            for i in range(len(input_total_part_cnt)):
                cnt = input_total_part_cnt[i]
                cur_parts = input_part_id[i]
                
                for j in range(cnt):
                    cur_part_id = cur_parts[j]
                    cur_part_pose = np.hstack([matched_pred_center2_all[t+j].cpu().numpy(), matched_pred_quat2_all[t+j].cpu().numpy()])
                    to_save[cur_part_id] = cur_part_pose
                fn = input_shape_id[t]+'-'+input_view_id[t]+'.npy'
                np.save(os.path.join(test_res_matched_dir , fn), to_save)
                print('saved', fn)
                t += cnt


        # get metric stats
        pred_pts = qrot(matched_pred_quat2_all.unsqueeze(1).repeat(1, num_point, 1), input_pts) + matched_pred_center2_all.unsqueeze(1).repeat(1, num_point, 1)
        gt_pts = qrot(matched_gt_quat_all.unsqueeze(1).repeat(1, num_point, 1), input_pts) + matched_gt_center_all.unsqueeze(1).repeat(1, num_point, 1)
        dist1, dist2 = chamfer_distance(gt_pts, pred_pts, transpose = False, sqrt=True)
        dist = torch.mean(dist1, dim=1).cpu().numpy() + torch.mean(dist2, dim=1).cpu().numpy()
        dist /= 2.0
        batch_size = input_box_size.shape[0]
        correct = 0
        cur_batch_accu = []
        cur_batch_correct = []
        t = 0
        for cnt in input_total_part_cnt:
            cur_shape_correct = 0
            for i in range(cnt):
                print('part dist' , dist[i], 'thresh' , eval_conf.thresh, 'correct', dist[i] < eval_conf.thresh)
                if dist[t+i] < eval_conf.thresh:
                    cur_shape_correct += 1
                    cur_batch_correct.append(True)
                else:
                    cur_batch_correct.append(False)
            cur_shape_accu = cur_shape_correct / cnt
            cur_batch_accu.append(cur_shape_accu)
            shape_accu.append(cur_shape_accu)
            t += cnt

        t = 0
        cur_batch_shape_chamfer = []
        for i in range(len(input_total_part_cnt)):
            cnt = input_total_part_cnt[i]
            gt_shape = gt_pts[t:t+cnt].view(1, -1, 3)
            pred_shape = pred_pts[t:t+cnt].view(1, -1, 3)
            ######## or ########
            dist1, dist2 = chamfer_distance(gt_shape, pred_shape, transpose = False, sqrt=True)
            dist = (torch.mean(dist1, dim=1).cpu().numpy()) + (torch.mean(dist2, dim=1).cpu().numpy())
            dist /= 2.0
            cur_batch_shape_chamfer.append(dist.mean())
            shape_chamfer.append(dist.mean())    
            t += cnt
 
        # visu
        if (not eval_conf.no_visu) and batch_id < eval_conf.visu_batch:
            print('Visualizing ...')
            t = 0
            for i in range(len(input_total_part_cnt)):
                cur_child_dir = os.path.join(visu_dir, 'child', 'data-%03d%03d' % (batch_id, r + i))
                os.mkdir(cur_child_dir)
                child_input_pts_dir = os.path.join(cur_child_dir, 'input_pts')
                os.mkdir(child_input_pts_dir)
                child_pred_mask_dir = os.path.join(cur_child_dir, 'pred_mask')
                os.mkdir(child_pred_mask_dir)
                child_gt_mask_dir = os.path.join(cur_child_dir, 'gt_mask')
                os.mkdir(child_gt_mask_dir)
                child_pred_pose_dir = os.path.join(cur_child_dir, 'pred_pose')
                os.mkdir(child_pred_pose_dir)
                child_pred_pose2_dir = os.path.join(cur_child_dir, 'pred_pose2')
                os.mkdir(child_pred_pose2_dir)
                child_gt_pose_dir = os.path.join(cur_child_dir, 'gt_pose')
                os.mkdir(child_gt_pose_dir)
                child_info_dir = os.path.join(cur_child_dir, 'info')
                os.mkdir(child_info_dir)
                fn = 'data-%03d%03d' % (batch_id, r + i)
                cur_input_img = (input_img[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(cur_input_img).save(os.path.join(input_img_dir, fn+'.png'))
                
                cnt = input_total_part_cnt[i]

                cur_shape_input_pts = input_pts[t:t+cnt].cpu().numpy()
                pred_shape_mask = matched_pred_mask_all[t:t+cnt].cpu().numpy()
                gt_shape_mask = matched_gt_mask_all[t:t+cnt].cpu().numpy()
                gt_mask_to_vis = np.ones((train_conf.img_size, train_conf.img_size, 3))
                pred_mask_to_vis = np.ones((train_conf.img_size, train_conf.img_size, 3))
                pred_pts = qrot(matched_pred_quat_all.unsqueeze(1).repeat(1, num_point, 1), input_pts) + matched_pred_center_all.unsqueeze(1).repeat(1, num_point, 1)
                gt_pts = qrot(matched_gt_quat_all.unsqueeze(1).repeat(1, num_point, 1), input_pts) + matched_gt_center_all.unsqueeze(1).repeat(1, num_point, 1)
                gt_shape_to_vis = gt_pts[t:t+cnt].view(-1, 1000, 3).cpu().numpy()
                pred_shape_to_vis = pred_pts[t:t+cnt].view(-1, 1000, 3).cpu().numpy()
                input_pts_to_vis = input_pts[t:t+cnt].view(-1, 1000, 3).cpu().numpy()


                pred_pts2 = qrot(matched_pred_quat2_all.unsqueeze(1).repeat(1, num_point, 1), input_pts) + matched_pred_center2_all.unsqueeze(1).repeat(1, num_point, 1)
                pred_shape_to_vis2 = pred_pts2[t:t+cnt].view(-1, 1000, 3).cpu().numpy()
                input_pts_to_vis = input_pts[t:t+cnt].view(-1, 1000, 3).cpu().numpy()

                for j in range(cnt):
                    # print(cnt)
                    cur_gt_shape_mask_to_vis = np.ones((train_conf.img_size, train_conf.img_size, 3))
                    cur_pred_shape_mask_to_vis = np.ones((train_conf.img_size, train_conf.img_size, 3))
                    child_fn = 'data-%03d%03d' % (batch_id, j)
                    if cnt > 20:
                        continue
                    color = colors[j] 
                    gt_inds = np.array(list(zip(*np.where(gt_shape_mask[j] > 0.5))))
                    pred_inds = np.array(list(zip(*np.where(pred_shape_mask[j] > 0.5))))
                    cur_pred_shape_mask_to_vis[:, :, 0] = (pred_shape_mask[j]*255)
                    cur_pred_shape_mask_to_vis[:, :, 1] = (pred_shape_mask[j]*255)
                    cur_pred_shape_mask_to_vis[:, :, 2] = (pred_shape_mask[j]*255)
                    if len(gt_inds) != 0:
                        gt_mask_to_vis[gt_inds[:, 0], gt_inds[:, 1], :] = (np.array(color))
                        cur_gt_shape_mask_to_vis[gt_inds[:, 0], gt_inds[:, 1], :] = (np.array(color))
                        cur_gt_shape_mask_to_vis *= 255
                        
                    if len(pred_inds) != 0:
                        pred_mask_to_vis[pred_inds[:,0], pred_inds[:,1], :] = (np.array(color))
                        # cur_pred_shape_mask_to_vis[pred_inds[:,0], pred_inds[:,1], :] = (np.array(color) * 255)
                        
                    cur_input_pts = cur_shape_input_pts[j]
                    render_utils.render_pts(os.path.join(BASE_DIR, child_input_pts_dir, child_fn), cur_input_pts, blender_fn='object_centered.blend')
                    cur_gt_mask = cur_gt_shape_mask_to_vis.astype(np.uint8)
                    Image.fromarray(cur_gt_mask).save(os.path.join(child_gt_mask_dir, child_fn+'.png'))
                    cur_pred_mask = cur_pred_shape_mask_to_vis.astype(np.uint8) 
                    Image.fromarray(cur_pred_mask).save(os.path.join(child_pred_mask_dir,  child_fn+'.png'))
                    cur_pred_pts = pred_shape_to_vis[j]
                    render_utils.render_pts(os.path.join(BASE_DIR, child_pred_pose_dir, child_fn), cur_pred_pts, blender_fn='camera_centered.blend')
                    cur_gt_pts = gt_shape_to_vis[j]
                    render_utils.render_pts(os.path.join(BASE_DIR, child_gt_pose_dir, child_fn), cur_gt_pts, blender_fn='camera_centered.blend')
                    cur_pred_pts2 = pred_shape_to_vis2[j]
                    render_utils.render_pts(os.path.join(BASE_DIR, child_pred_pose2_dir, child_fn), cur_pred_pts2, blender_fn='camera_centered.blend')
                    
                    with open(os.path.join(child_info_dir, child_fn + '.txt'), 'w') as f:
                        f.write('mask_loss: %f\n' % mask_loss_per_data_all[i][j].item())
                        f.write('center_loss: %f\n' % center_loss_per_data[t+j].item())
                        f.write('quat_loss: %f\n' % quat_loss_per_data[t+j].item())
                        f.write('pose_loss: %f\n' % pose_loss_per_data[t+j].item())
                        f.write('box_loss: %f\n' % box_loss_per_data[t+j].item())
                        f.write('anchor_loss: %f\n' % anchor_loss_per_data[t+j].item())
                        f.write('l2_rotation_loss: %f\n' % l2_rotation_loss_per_data[t+j].item())
                        f.write('l2_joint_loss: %f\n' % l2_joint_loss_per_data[t+j].item())
                        f.write('ins_onehot: %s\n' % matched_ins_onehot_all[t+j])
                        f.write('correct pose: %s\n' % ( str(cur_batch_correct[t+j]) ) ) 
                render_utils.render_part_pts(os.path.join(BASE_DIR, pred_dof_dir, fn), pred_shape_to_vis, blender_fn='camera_centered.blend')
                render_utils.render_part_pts(os.path.join(BASE_DIR, pred_dof2_dir, fn), pred_shape_to_vis2, blender_fn='camera_centered.blend')
                render_utils.render_part_pts(os.path.join(BASE_DIR, gt_dof_dir, fn), gt_shape_to_vis, blender_fn='camera_centered.blend')
                render_utils.render_part_pts(os.path.join(BASE_DIR, input_pts_dir, fn), input_pts_to_vis, blender_fn='object_centered.blend')
                gt_mask_to_vis *= 255
                Image.fromarray(gt_mask_to_vis.astype(np.uint8) ).save(os.path.join(gt_mask_dir, fn+'.png'))
                pred_mask_to_vis *= 255
                Image.fromarray(pred_mask_to_vis.astype(np.uint8) ).save(os.path.join(pred_mask_dir, fn+'.png'))

                with open(os.path.join(info_dir, fn + '.txt'), 'w') as fout:
                    fout.write('shape_id: %s, view_id: %s\n' % (\
                            batch[data_features.index('shape_id')][i], \
                            batch[data_features.index('view_id')][i]))
                    fout.write('mask_loss: %f\n' % torch.sum(mask_loss_per_data_all[i]).item())
                    fout.write('center_loss: %f\n' % torch.sum(center_loss_per_data[t:t+cnt]).item())
                    fout.write('quat_loss: %f\n' % torch.sum(quat_loss_per_data[t:t+cnt]).item())  
                    fout.write('pose_loss: %f\n' % torch.sum(pose_loss_per_data[t:t+cnt]).item())
                    fout.write('box_loss: %f\n' % torch.sum(box_loss_per_data[t:t+cnt]).item())
                    fout.write('anchor_loss: %f\n' % torch.sum(anchor_loss_per_data[t:t+cnt]).item())
                    fout.write('l2_rotation_loss: %f\n' % torch.sum(l2_rotation_loss_per_data[t:t+cnt]).item())
                    fout.write('l2_joint_loss: %f\n' % torch.sum(l2_joint_loss_per_data[t:t+cnt]).item())
                    fout.write('whole_shape_cd_loss: %f\n' % whole_shape_cd_per_shape[i].item()) 
                    fout.write('cur shape accu: %f\n' % cur_batch_accu[i])
                    fout.write('shape chamfer: %f \n ' % cur_batch_shape_chamfer[i])

                t += cnt
            # visu a html
            if batch_id == eval_conf.visu_batch - 1:
                print('Generating html visualization ...')
                sublist = 'input_img,input_pts,gt_mask,pred_mask,gt_assembly,pred_assembly,pred_assembly2,info:gt_mask,pred_mask,gt_pose,pred_pose,pred_pose2,info'
                cmd = 'cd %s && python %s . 1 htmls %s %s > /dev/null' % (visu_dir, os.path.join(BASE_DIR, '../utils/gen_html_hierachy_local.py'), sublist, sublist)
                call(cmd, shell=True)
                print('DONE')



print('shape accuracy', shape_accu)
shape_accu_final = np.array(shape_accu).mean()
print('shape chamfer', shape_chamfer)
shape_chamfer_final = np.array(shape_chamfer).mean()


print(' from ckpoints ', os.path.join('logs', eval_conf.exp_name, 'ckpts'), eval_conf.model_epoch)

print('             *** part accuracy ***              ', shape_accu_final)
print('             *** shape chamfer ***              ', shape_chamfer_final)
