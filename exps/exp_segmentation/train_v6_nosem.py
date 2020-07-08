"""
    For models: model_v2.py, etc. (GRU)
"""

import os
import time
import sys
import shutil
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
from PIL import Image
from subprocess import call
from data import PartNetShapeDataset
import utils
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import render_using_blender as render_utils
from quaternion import qrot


def train(conf):
    # create training and validation datasets and data loaders
    data_features = ['img', 'pts', 'ins_one_hot' , 'box_size', 'total_parts_cnt' , 'similar_parts_cnt', 'mask' ,'shape_id', 'view_id']
    
    train_dataset = PartNetShapeDataset(conf.category, conf.data_dir, data_features, data_split="train", \
            max_num_mask = conf.max_num_parts, max_num_similar_parts=conf.max_num_similar_parts, img_size=conf.img_size, on_kaichun_machine=conf.on_kaichun_machine)
    utils.printout(conf.flog, str(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, pin_memory=True, \
            num_workers=conf.num_workers, drop_last=True, collate_fn=utils.collate_feats_with_none, worker_init_fn=utils.worker_init_fn)
    
    val_dataset = PartNetShapeDataset(conf.category, conf.data_dir, data_features, data_split="val", \
            max_num_mask = conf.max_num_parts, max_num_similar_parts=conf.max_num_similar_parts, img_size=conf.img_size, on_kaichun_machine=conf.on_kaichun_machine)
    utils.printout(conf.flog, str(val_dataset))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=False, pin_memory=True, \
            num_workers=0, drop_last=True, collate_fn=utils.collate_feats_with_none, worker_init_fn=utils.worker_init_fn)

    # load network model
    model_def = utils.get_model_module(conf.model_version)

    # create models
    network = model_def.Network(conf, train_dataset.get_part_count())
    utils.printout(conf.flog, '\n' + str(network) + '\n')

    models = [network]
    model_names = ['network']

    # create optimizers
    network_opt = torch.optim.Adam(network.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    optimizers = [network_opt]
    optimizer_names = ['network_opt']

    # learning rate scheduler
    network_lr_scheduler = torch.optim.lr_scheduler.StepLR(network_opt, step_size=conf.lr_decay_every, gamma=conf.lr_decay_by)

    # create logs
    if not conf.no_console_log:
        header = '     Time    Epoch     Dataset    Iteration    Progress(%)       LR    CenterLoss    QuatLoss   TotalLoss'
    if not conf.no_tb_log:
        # https://github.com/lanpa/tensorboard-pytorch
        from tensorboardX import SummaryWriter
        train_writer = SummaryWriter(os.path.join(conf.exp_dir, 'train'))
        val_writer = SummaryWriter(os.path.join(conf.exp_dir, 'val'))

    # send parameters to device
    for m in models:
        m.to(conf.device)
    for o in optimizers:
        utils.optimizer_to_device(o, conf.device)

    # start training
    start_time = time.time()

    last_checkpoint_step = None
    last_train_console_log_step, last_val_console_log_step = None, None
    train_num_batch = len(train_dataloader)
    val_num_batch = len(val_dataloader)

    # train for every epoch
    for epoch in range(conf.epochs):
        if not conf.no_console_log:
            utils.printout(conf.flog, f'training run {conf.exp_name}')
            utils.printout(conf.flog, header)

        train_batches = enumerate(train_dataloader, 0)
        val_batches = enumerate(val_dataloader, 0)
        train_fraction_done = 0.0
        val_fraction_done = 0.0
        val_batch_ind = -1

        # train for every batch
        for train_batch_ind, batch in train_batches:
            train_fraction_done = (train_batch_ind + 1) / train_num_batch
            train_step = epoch * train_num_batch + train_batch_ind

            log_console = not conf.no_console_log and (last_train_console_log_step is None or \
                    train_step - last_train_console_log_step >= conf.console_log_interval)
            if log_console:
                last_train_console_log_step = train_step

            # set models to training mode
            for m in models:
                m.train()

            # forward pass (including logging)
            total_loss = forward(batch=batch, data_features=data_features, network=network, conf=conf, is_val=False, \
                    step=train_step, epoch=epoch, batch_ind=train_batch_ind, num_batch=train_num_batch, start_time=start_time, \
                    log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=train_writer, lr=network_opt.param_groups[0]['lr'])

            if total_loss is not None:
                # optimize one step
                network_lr_scheduler.step()
                network_opt.zero_grad()
                total_loss.backward()
                network_opt.step()

            # save checkpoint
            with torch.no_grad():
                if last_checkpoint_step is None or train_step - last_checkpoint_step >= conf.checkpoint_interval:
                    utils.printout(conf.flog, 'Saving checkpoint ...... ')
                    utils.save_checkpoint(models=models, model_names=model_names, dirname=os.path.join(conf.exp_dir, 'ckpts'), \
                            epoch=epoch, prepend_epoch=True, optimizers=optimizers, optimizer_names=model_names)
                    utils.printout(conf.flog, 'DONE')
                    last_checkpoint_step = train_step

            # validate one batch
            while val_fraction_done <= train_fraction_done and val_batch_ind+1 < val_num_batch:
                val_batch_ind, val_batch = next(val_batches)

                val_fraction_done = (val_batch_ind + 1) / val_num_batch
                val_step = (epoch + val_fraction_done) * train_num_batch - 1

                log_console = not conf.no_console_log and (last_val_console_log_step is None or \
                        val_step - last_val_console_log_step >= conf.console_log_interval)
                if log_console:
                    last_val_console_log_step = val_step

                # set models to evaluation mode
                for m in models:
                    m.eval()

                with torch.no_grad():
                    # forward pass (including logging)
                    __ = forward(batch=val_batch, data_features=data_features, network=network, conf=conf, is_val=True, \
                            step=val_step, epoch=epoch, batch_ind=val_batch_ind, num_batch=val_num_batch, start_time=start_time, \
                            log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=val_writer, lr=network_opt.param_groups[0]['lr'])
           
    # save the final models
    utils.printout(conf.flog, 'Saving final checkpoint ...... ')
    utils.save_checkpoint(models=models, model_names=model_names, dirname=os.path.join(conf.exp_dir, 'ckpts'), \
            epoch=epoch, prepend_epoch=False, optimizers=optimizers, optimizer_names=optimizer_names)
    utils.printout(conf.flog, 'DONE')


def forward(batch, data_features, network, conf, \
        is_val=False, step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0, \
        log_console=False, log_tb=False, tb_writer=None, lr=None):
    # prepare input
    # generate a batch of data size  < 64
    batch_index = 1
    if len(batch) == 0:
        return None

    cur_batch_size = len(batch[data_features.index('total_parts_cnt')])
    total_part_cnt = batch[data_features.index('total_parts_cnt')][0]

    if total_part_cnt == 1:
        print('passed an entire shape does not work for batch norm')
        return None
    input_total_part_cnt = batch[data_features.index('total_parts_cnt')][0]                             # 1
    input_img = batch[data_features.index('img')][0]                                                    # 3 x H x W
    input_img = input_img.repeat(input_total_part_cnt, 1, 1, 1)                            # part_cnt 3 x H x W
    input_pts = batch[data_features.index('pts')][0].squeeze(0)[:input_total_part_cnt]                             # part_cnt x N x 3
    # input_sem_one_hot = batch[data_features.index('sem_one_hot')][0].squeeze(0)[:input_total_part_cnt]             # part_cnt x K
    input_ins_one_hot = batch[data_features.index('ins_one_hot')][0].squeeze(0)[:input_total_part_cnt]             # part_cnt x max_similar_parts
    input_similar_part_cnt = batch[data_features.index('similar_parts_cnt')][0].squeeze(0)[:input_total_part_cnt]  # part_cnt x 1    
    input_box_size = batch[data_features.index('box_size')][0].squeeze(0)[:input_total_part_cnt]

    # prepare gt: 
    gt_mask = (batch[data_features.index('mask')][0].squeeze(0)[:input_total_part_cnt].to(conf.device),)  
    input_total_part_cnt = [batch[data_features.index('total_parts_cnt')][0]]
    while total_part_cnt < 32 and batch_index < cur_batch_size:
        cur_input_cnt = batch[data_features.index('total_parts_cnt')][batch_index]
        total_part_cnt += cur_input_cnt
        if total_part_cnt > 40:
            total_part_cnt -= cur_input_cnt
            batch_index += 1
            continue
        cur_batch_img = batch[data_features.index('img')][batch_index].repeat(cur_input_cnt, 1, 1, 1)
        input_img = torch.cat((input_img, cur_batch_img), dim=0)
        cur_box_size = batch[data_features.index('box_size')][batch_index].squeeze(0)[:cur_input_cnt]
        input_box_size = torch.cat( (input_box_size, cur_box_size), dim=0)   
        input_pts = torch.cat((input_pts, batch[data_features.index('pts')][batch_index].squeeze(0)[:cur_input_cnt]), dim=0)                            # B x max_parts x N x 3
        # input_sem_one_hot = torch.cat((input_sem_one_hot, batch[data_features.index('sem_one_hot')][batch_index].squeeze(0)[:cur_input_cnt]), dim=0)    # B x max_parts x K
        input_ins_one_hot = torch.cat((input_ins_one_hot, batch[data_features.index('ins_one_hot')][batch_index].squeeze(0)[:cur_input_cnt]), dim=0)    # B x max_parts x max_similar_parts
        input_total_part_cnt.append(batch[data_features.index('total_parts_cnt')][batch_index])                             # 1
        input_similar_part_cnt = torch.cat((input_similar_part_cnt, batch[data_features.index('similar_parts_cnt')][batch_index].squeeze(0)[:cur_input_cnt]), dim=0)  # B x max_parts x 2    
        # prepare gt
        gt_mask = gt_mask + (batch[data_features.index('mask')][batch_index].squeeze(0)[:cur_input_cnt].to(conf.device), )
        batch_index += 1

    input_img = input_img.to(conf.device); input_pts = input_pts.to(conf.device); # input_sem_one_hot = input_sem_one_hot.to(conf.device); 
    input_similar_part_cnt = input_similar_part_cnt.to(conf.device); input_ins_one_hot = input_ins_one_hot.to(conf.device)
    input_box_size = input_box_size.to(conf.device)
    batch_size = input_img.shape[0]
    num_point = input_pts.shape[1]
    #num_sem = input_sem_one_hot.shape[1]

    # forward through the network
    pred_masks = network(input_img - 0.5, input_pts, input_ins_one_hot, input_box_size, input_total_part_cnt)
    
    # perform matching and calculate masks 
    mask_loss_per_data = []; t = 0;
    matched_pred_mask_all = torch.zeros(batch_size, 224, 224); matched_gt_mask_all = torch.zeros(batch_size, 224, 224) 
    for i in range(len(input_total_part_cnt)):
        total_cnt = input_total_part_cnt[i]
        matched_gt_ids, matched_pred_ids = network.linear_assignment(gt_mask[i], pred_masks[i][:-1, :,:], input_similar_part_cnt[t:t+total_cnt])
        

        # select the matched data
        matched_pred_mask = pred_masks[i][matched_pred_ids]
        matched_gt_mask = gt_mask[i][matched_gt_ids]

        matched_gt_mask_all[t:t+total_cnt, :, :] = matched_gt_mask
        matched_pred_mask_all[t:t+total_cnt, :, :] = matched_pred_mask

        # for computing mask soft iou loss
        matched_mask_loss = network.get_mask_loss(matched_pred_mask, matched_gt_mask)

        mask_loss_per_data.append(matched_mask_loss.mean())
        t+= total_cnt
    mask_loss_per_data = torch.stack(mask_loss_per_data)
    
    # for each type of loss, compute avg loss per batch
    mask_loss = mask_loss_per_data.mean()

    # compute total loss
    total_loss = mask_loss * conf.loss_weight_mask

    # display information
    data_split = 'train'
    if is_val:
        data_split = 'val'

    with torch.no_grad():
        # log to console
        if log_console:
            utils.printout(conf.flog, \
                f'''{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} '''
                f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                f'''{data_split:^10s} '''
                f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                f'''{100. * (1+batch_ind+num_batch*epoch) / (num_batch*conf.epochs):>9.1f}%      '''
                f'''{lr:>5.2E} '''
                f'''{mask_loss.item():>10.5f}'''
                f'''{total_loss.item():>10.5f}''')
            conf.flog.flush()

        # log to tensorboard
        if log_tb and tb_writer is not None:
            tb_writer.add_scalar('mask_loss', mask_loss.item(), step)
            tb_writer.add_scalar('total_loss', total_loss.item(), step)
            tb_writer.add_scalar('lr', lr, step)

        # gen visu
        if is_val and (not conf.no_visu) and epoch % conf.num_epoch_every_visu == 0:
            visu_dir = os.path.join(conf.exp_dir, 'val_visu')
            out_dir = os.path.join(visu_dir, 'epoch-%04d' % epoch)
            input_img_dir = os.path.join(out_dir, 'input_img')
            input_pts_dir = os.path.join(out_dir, 'input_pts')
            gt_mask_dir = os.path.join(out_dir, 'gt_mask')
            pred_mask_dir = os.path.join(out_dir, 'pred_mask')
            info_dir = os.path.join(out_dir, 'info')

            if batch_ind == 0:
                # create folders
                os.mkdir(out_dir)
                os.mkdir(input_img_dir)
                os.mkdir(input_pts_dir)
                os.mkdir(gt_mask_dir)
                os.mkdir(pred_mask_dir)
                os.mkdir(info_dir)

            if batch_ind < conf.num_batch_every_visu:
                utils.printout(conf.flog, 'Visualizing ...')

                t = 0
                for i in range(batch_size):
                    fn = 'data-%03d.png' % (batch_ind * batch_size + i)

                    cur_input_img = (input_img[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(cur_input_img).save(os.path.join(input_img_dir, fn))
                    cur_input_pts = input_pts[i].cpu().numpy()
                    render_utils.render_pts(os.path.join(BASE_DIR, input_pts_dir, fn), cur_input_pts, blender_fn='object_centered.blend')
                    cur_gt_mask = (matched_gt_mask_all[i].cpu().numpy() > 0.5).astype(np.uint8) * 255
                    Image.fromarray(cur_gt_mask).save(os.path.join(gt_mask_dir, fn))
                    cur_pred_mask = (matched_pred_mask_all[i].cpu().numpy() > 0.5).astype(np.uint8) * 255
                    Image.fromarray(cur_pred_mask).save(os.path.join(pred_mask_dir, fn))
                    

                    #with open(os.path.join(info_dir, fn.replace('.png', '.txt')), 'w') as fout:
                        # fout.write('shape_id: %s, view_id: %s\n' % (\
                                # batch[data_features.index('shape_id')][i],\
                                # batch[data_features.index('view_id')][i]))
                        #fout.write('ins_one_hot: %s\n sem_one_hot: %s\n' % (str(input_ins_one_hot[i]), str(input_sem_one_hot[i])))   
                
            if batch_ind == conf.num_batch_every_visu - 1:
                # visu html
                utils.printout(conf.flog, 'Generating html visualization ...')
                sublist = 'input_img,input_pts,gt_mask,pred_mask,info'
                cmd = 'cd %s && python %s . 10 htmls %s %s > /dev/null' % (out_dir, os.path.join(BASE_DIR, '../utils/gen_html_hierachy_local.py'), sublist, sublist)
                call(cmd, shell=True)
                utils.printout(conf.flog, 'DONE')

    return total_loss


if __name__ == '__main__':
    ### get parameters
    parser = ArgumentParser()
    
    # main parameters (required)
    parser.add_argument('--exp_suffix', type=str, help='exp suffix')
    parser.add_argument('--model_version', type=str, help='model def file')
    parser.add_argument('--category', type=str, help='model def file')
    parser.add_argument('--train_data_fn', type=str, help='training data file that indexs all data tuples')
    parser.add_argument('--val_data_fn', type=str, help='validation data file that indexs all data tuples')

    # main parameters (optional)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    #parser.add_argument('--seed', type=int, default=3124256514, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--log_dir', type=str, default='logs', help='exp logs directory')
    parser.add_argument('--data_dir', type=str, default='../../data/final-semzX/', help='data directory')
    parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if exp_dir exists [default: False]')
    parser.add_argument('--on_kaichun_machine', action='store_true', default=False, help='on kaichun machine? [default: False]')

    # network settings
    parser.add_argument('--pointnet_emd_dim', type=int, default=512)
    parser.add_argument('--max_num_parts', type=int, default=20)
    parser.add_argument('--max_num_similar_parts', type=int, default=21)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--use_semantics', action='store_true', default=False)
    parser.add_argument('--pretrain_resnet', action='store_true', default=False)
    parser.add_argument('--normalize', action='store_true', default=False)


    # training parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_decay_by', type=float, default=0.9)
    parser.add_argument('--lr_decay_every', type=float, default=5000)

    # loss weights
    parser.add_argument('--loss_weight_mask', type=float, default=1.0, help='loss weight')

    # logging
    parser.add_argument('--no_tb_log', action='store_true', default=False)
    parser.add_argument('--no_console_log', action='store_true', default=False)
    parser.add_argument('--console_log_interval', type=int, default=10, help='number of optimization steps beween console log prints')
    parser.add_argument('--checkpoint_interval', type=int, default=10000, help='number of optimization steps beween checkpoints')

    # visu
    parser.add_argument('--num_batch_every_visu', type=int, default=1, help='num batch every visu')
    parser.add_argument('--num_epoch_every_visu', type=int, default=1, help='num epoch every visu')
    parser.add_argument('--no_visu', action='store_true', default=False, help='no visu? [default: False]')

    # parse args
    conf = parser.parse_args()


    ### prepare before training
    # make exp_name
    conf.exp_name = f'exp-{conf.category}-{conf.model_version}-{conf.exp_suffix}'
    
    # mkdir exp_dir; ask for overwrite if necessary
    conf.exp_dir = os.path.join(conf.log_dir, conf.exp_name)
    if os.path.exists(conf.exp_dir):
        if not conf.overwrite:
            response = input('A training run named "%s" already exists, overwrite? (y/n) ' % conf.exp_name)
            if response != 'y':
                exit(1)
        shutil.rmtree(conf.exp_dir)
    os.mkdir(conf.exp_dir)
    os.mkdir(os.path.join(conf.exp_dir, 'ckpts'))
    if not conf.no_visu:
        os.mkdir(os.path.join(conf.exp_dir, 'val_visu'))

    # control randomness
    if conf.seed < 0:
        conf.seed = random.randint(1, 10000)
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)

    # save config
    torch.save(conf, os.path.join(conf.exp_dir, 'conf.pth'))

    # file log
    flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'w')
    conf.flog = flog

    # backup command running
    utils.printout(flog, ' '.join(sys.argv) + '\n')
    utils.printout(flog, f'Random Seed: {conf.seed}')

    # backup python files used for this training
    os.system('cp data.py models/%s.py %s %s' % (conf.model_version, __file__, conf.exp_dir))
     
    # set training device
    device = torch.device(conf.device)
    utils.printout(flog, f'Using device: {conf.device}\n')
    conf.device = device

    # set the max num mask to max num part
    conf.max_num_mask = conf.max_num_parts
    conf.ins_dim = conf.max_num_similar_parts

    ### start training
    train(conf)


    ### before quit
    # close file log
    flog.close()

