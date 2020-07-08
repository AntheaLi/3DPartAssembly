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
from data_real import RealData
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
parser.add_argument('--data_dir', type=str, default='/home/kaichun/part_assembly/data/real/', help='test data fn')
parser.add_argument('--result_suffix', type=str, default='nothing')
parser.add_argument('--model_epoch', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--visu_batch', type=int, default=1)
parser.add_argument('--thresh', type=float, default=0.05)
parser.add_argument('--save', action='store_true', default=False, help='save results? [default: False]')
parser.add_argument('--no_visu', action='store_true', default=False, help='no visu? [default: False]')
parser.add_argument('--log_dir', type=str, default='logs', help='exp logs directory')
parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
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
    pred_mask_dir = os.path.join(visu_dir, 'pred_mask')
    os.mkdir(pred_mask_dir)
    pred_dof_dir = os.path.join(visu_dir, 'pred_assembly')
    os.mkdir(pred_dof_dir)
    pred_dof2_dir = os.path.join(visu_dir, 'pred_assembly2')
    os.mkdir(pred_dof2_dir)
    child_dir = os.path.join(visu_dir, 'child')
    os.mkdir(child_dir)
    info_dir = os.path.join(visu_dir, 'info')
    os.mkdir(info_dir)


# dataset
data_features = ['img', 'pts', 'total_parts_cnt' ,'ins_one_hot' , 'sem_one_hot', 'similar_parts_cnt', 'similar_parts_edge_indices']
dataset = RealData(eval_conf.data_dir, data_features) 
print(dataset, len(dataset))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=eval_conf.batch_size, shuffle=False, pin_memory=True, \
        num_workers=0, drop_last=False, collate_fn=utils.collate_feats, worker_init_fn=utils.worker_init_fn)
 
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

        cur_batch_size = len(batch[data_features.index('total_parts_cnt')])
        total_part_cnt = batch[data_features.index('total_parts_cnt')][0]
        input_total_part_cnt = batch[data_features.index('total_parts_cnt')][0]                             # 1
        input_img = batch[data_features.index('img')][0]                                                    # 3 x H x W
        input_img = input_img.repeat(input_total_part_cnt, 1, 1, 1)                            # part_cnt 3 x H x W
        input_pts = batch[data_features.index('pts')][0].squeeze(0)[:input_total_part_cnt]                             # part_cnt x N x 3
        input_sem_one_hot = batch[data_features.index('sem_one_hot')][0].squeeze(0)[:input_total_part_cnt]             # part_cnt x K
        input_ins_one_hot = batch[data_features.index('ins_one_hot')][0].squeeze(0)[:input_total_part_cnt]             # part_cnt x max_similar_parts
        input_similar_part_cnt = batch[data_features.index('similar_parts_cnt')][0].squeeze(0)[:input_total_part_cnt]  # part_cnt x 1    
    
        # prepare gt: 
        input_total_part_cnt = [batch[data_features.index('total_parts_cnt')][0]]
        input_similar_parts_edge_indices = [batch[data_features.index('similar_parts_edge_indices')][0].to(train_conf.device)]
        
        input_img = input_img.to(train_conf.device); input_pts = input_pts.to(train_conf.device); input_sem_one_hot = input_sem_one_hot.to(train_conf.device); 
        input_similar_part_cnt = input_similar_part_cnt.to(train_conf.device); input_ins_one_hot = input_ins_one_hot.to(train_conf.device)
               

        # prepare gt
        batch_size = 1
        num_point = 1000


        # forward through the network
        pred_masks, pred_center, pred_quat, pred_center2, pred_quat2 = network(input_img - 0.5, input_pts, input_sem_one_hot, input_ins_one_hot,  input_total_part_cnt, input_similar_parts_edge_indices)
                
        part_order = ['3', '0_0', '0_1', '1_0', '1_1', '2_0', '2_1', '2_2', '2_3']

        to_save = {}
        for part in part_order:
            quat1 = pred_quat.cpu().numpy()
            center1 = pred_center.cpu().numpy()
            quat2 = pred_quat2.cpu().numpy()
            center2 = pred_center2.cpu().numpy()    
            to_save[part + 'quat1'] = quat1
            to_save[part + 'quat2'] = quat2
            to_save[part + 'center1'] = center1
            to_save[part + 'center2'] = center2
        np.save( test_res_dir + 'res.npy' , to_save)
 
        # visu
        if (not eval_conf.no_visu) and batch_id < eval_conf.visu_batch:
            print('Visualizing ...')
            t = 0
            for i in range(len(input_total_part_cnt)):
                cur_child_dir = os.path.join(visu_dir, 'child', 'data-%03d%03d' % (batch_id, r + i))
                os.mkdir(cur_child_dir)
                child_input_pts_dir = os.path.join(cur_child_dir, 'input_pts')
                os.mkdir(child_input_pts_dir)
                child_pred_pose_dir = os.path.join(cur_child_dir, 'pred_pose')
                os.mkdir(child_pred_pose_dir)
                child_pred_pose2_dir = os.path.join(cur_child_dir, 'pred_pose2')
                os.mkdir(child_pred_pose2_dir)
                fn = 'data-%03d%03d' % (batch_id, r + i)
                cur_input_img = (input_img[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(cur_input_img).save(os.path.join(input_img_dir, fn+'.png'))
                
                cnt = input_total_part_cnt[i]

                cur_shape_input_pts = input_pts[t:t+cnt].cpu().numpy()
                pred_pts = qrot(pred_quat.unsqueeze(1).repeat(1, num_point, 1), input_pts) + pred_center.unsqueeze(1).repeat(1, num_point, 1)
                pred_shape_to_vis = pred_pts[t:t+cnt].view(-1, 1000, 3).cpu().numpy()
                input_pts_to_vis = input_pts[t:t+cnt].view(-1, 1000, 3).cpu().numpy()


                pred_pts2 = qrot(pred_quat2.unsqueeze(1).repeat(1, num_point, 1), input_pts) + pred_center2.unsqueeze(1).repeat(1, num_point, 1)
                pred_shape_to_vis2 = pred_pts2[t:t+cnt].view(-1, 1000, 3).cpu().numpy()
                input_pts_to_vis = input_pts[t:t+cnt].view(-1, 1000, 3).cpu().numpy()

                for j in range(cnt):
                    child_fn = 'data-%03d%03d' % (batch_id, j)
                    
                    color = colors[j] 
                        
                    cur_input_pts = cur_shape_input_pts[j]
                    render_utils.render_pts(os.path.join(BASE_DIR, child_input_pts_dir, child_fn), cur_input_pts, blender_fn='object_centered.blend')
                    cur_pred_pts = pred_shape_to_vis[j]
                    render_utils.render_pts(os.path.join(BASE_DIR, child_pred_pose_dir, child_fn), cur_pred_pts, blender_fn='camera_centered.blend')
                    cur_pred_pts2 = pred_shape_to_vis2[j]
                    render_utils.render_pts(os.path.join(BASE_DIR, child_pred_pose2_dir, child_fn), cur_pred_pts2, blender_fn='camera_centered.blend')
                    
                render_utils.render_part_pts(os.path.join(BASE_DIR, pred_dof_dir, fn), pred_shape_to_vis, blender_fn='camera_centered.blend')
                render_utils.render_part_pts(os.path.join(BASE_DIR, pred_dof2_dir, fn), pred_shape_to_vis2, blender_fn='camera_centered.blend')
                render_utils.render_part_pts(os.path.join(BASE_DIR, input_pts_dir, fn), input_pts_to_vis, blender_fn='object_centered.blend')

                t += cnt
            # visu a html
            if batch_id == eval_conf.visu_batch - 1:
                print('Generating html visualization ...')
                sublist = 'input_img,input_pts,gt_mask,pred_mask,gt_assembly,pred_assembly,pred_assembly2,info:gt_mask,pred_mask,gt_pose,pred_pose,pred_pose2,info'
                cmd = 'cd %s && python %s . 1 htmls %s %s > /dev/null' % (visu_dir, os.path.join(BASE_DIR, '../utils/gen_html_hierachy_local.py'), sublist, sublist)
                call(cmd, shell=True)
                print('DONE')

