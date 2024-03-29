"""
    PartNetPartDataset
"""

import os, sys
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from quaternion import qrot


class PartNetShapeDataset(data.Dataset):

    def __init__(self, category, data_dir, data_features, data_split='train', \
            max_num_mask=150 , max_num_similar_parts=10, num_points = 1000, img_size=224):
        # store parameters
        self.data_dir = data_dir
        self.data_split = data_split
        self.category = category
        self.img_size = img_size
        self.max_num_similar_parts = max_num_similar_parts
        self.max_num_mask = max_num_mask
        self.num_points = num_points
        self.img_path = '/orion/group/PartNet/partnet_rgb_masks_{}'.format(self.category.split('-')[0].lower())
        
        # load data
        self.data = []
        if 'Chair-mixed' in self.category or'Table-mixed' in self.category:
            self.data_path = os.path.join('../../stats/train_test_splits/{}-{}.txt'.format(self.category[:11], self.data_split))
        elif 'Chair-3' in self.category or'Table-3' in self.category:
            self.data_path = os.path.join('../../stats/train_test_splits/{}-{}.txt'.format(self.category[:7], self.data_split))
        elif 'StorageFurniture-3' in self.category:
            self.data_path = os.path.join('../../stats/train_test_splits/{}-{}.txt'.format(self.category[:18], self.data_split))
        else:
            self.data_path = os.path.join('../../stats/train_test_splits/{}-{}.txt'.format(self.category[:22], self.data_split))

        with open(self.data_path, 'r') as f:
            for line in f:
                self.data.append(line.strip())

        self.data_features = data_features

        # load category semantics information
        self.part_sems = []
        self.part_sem2id = dict()
        if not 'mixed' in self.category:
            with open(os.path.join('../../stats/part_semantics/', self.category.split('-')[0]+'-level-'+ self.category.split('-')[1]+'.txt'), 'r') as fin:
                for i, l in enumerate(fin.readlines()):
                    _, part_sem, _ = l.rstrip().split()
                    self.part_sems.append(part_sem)
                    self.part_sem2id[part_sem] = i


    def get_part_count(self):
        return len(self.part_sems)
        
    def convert_view(self, n):
        return '%02d'%n

    def __str__(self):
        strout = '[PartNetPartDataset %s %d] data_dir: %s, data_split: %s, max_num_similar_parts: %d' % \
                (self.category, len(self), self.data_dir, self.data_split, self.max_num_similar_parts)
        return strout

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        shape_id = self.data[index]
        cur_data_fn = os.path.join(self.data_dir, self.category, '%s.npy' % (shape_id))
        cur_data = np.load(cur_data_fn, allow_pickle=True).item()
        view_id = self.convert_view(np.random.choice(np.arange(24)))
        if self.data_split == "val" or self.data_split=="vis" or self.data_split == "test":
            view_id = "00"
        cur_data_view_fn = os.path.join(self.data_dir, self.category, shape_id, '%s.npy' % (view_id))
        cur_data_view = np.load(cur_data_view_fn, allow_pickle=True).item()
        part_id = np.random.choice(cur_data['all_parts'])

        equiv_edge_indices = []

        # sort part order to put the single part in front and similar parts in the back
        single_parts = [x for x in cur_data['all_parts'] if len(cur_data[x]['similar']) == 1]
        # similar_part_inds = [i+1 for i in range(len(single_parts))]
        other_parts = [x for x in cur_data['all_parts'] if len(cur_data[x]['similar']) > 1]
        part_order = sorted(single_parts)
        t = len(part_order)
        for part in other_parts:
            if part not in part_order:
                cur_cnt = len(cur_data[part]['similar'])
                part_order+=sorted(cur_data[part]['similar'])
                for t1 in range(t, t+cur_cnt):
                    for t2 in range(t, t+cur_cnt):
                        if t1 != t2:
                            equiv_edge_indices.append([t1, t2])
                t += cur_cnt

        if len(part_order) != len(cur_data['all_parts']):
            print('check data ', shape_id, view_id)

        
        data_feats = ()
        for feat in self.data_features:
            if feat == 'img':
                img_fn = os.path.join(self.img_path, shape_id, 'view-{}'.format(view_id), 'shape-rgb.png')
                if not os.path.exists(img_fn):
                    return None
                with Image.open(img_fn) as fimg:
                    out = np.array(fimg, dtype=np.float32) / 255
                white_img = np.ones((self.img_size, self.img_size, 3), dtype=np.float32)
                mask = np.tile(out[:, :, 3:4], [1, 1, 3])
            
                out = out[:, :, :3] * mask + white_img * (1 - mask)
                out = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
                data_feats = data_feats + (out,)
            
            elif feat == 'mask':
                out = np.zeros((self.max_num_mask, self.img_size, self.img_size), dtype=np.float32)
                if len(cur_data['all_parts']) > self.max_num_mask:
                    return None
                if len(cur_data['all_parts']) <= 1:
                    return None
                for i in range(len(part_order)):
                    part_id = part_order[i]
                    mask_ids = np.array(cur_data_view[part_id]['mask'])
                    if len(mask_ids) > 0:
                        out[i, mask_ids[:, 0], mask_ids[:, 1]] = 1 # real mask starts from [:, 1:, 1:] of the tensor 
                out = torch.from_numpy(out).unsqueeze(0)
                data_feats = data_feats + (out,) # output tenshor shape [1, max mask, 225, 225]


            elif feat == 'pts':
                out = torch.zeros(self.max_num_mask, self.num_points, 3 ).float()  # first-dim is 0/1, meaning exist or not
                if len(cur_data['all_parts']) > self.max_num_mask:
                    return None
                if len(cur_data['all_parts']) <= 1:
                    return None
                for i in range(len(part_order)):
                    part_id = part_order[i]
                    if len(cur_data[part_id]['similar']) > self.max_num_similar_parts:
                        return None
                    out[i, :, :] = torch.from_numpy(cur_data[part_id]['pts']).float()
                out = out.unsqueeze(0)
                data_feats = data_feats + (out,)


            elif feat == 'box_size':
                out = torch.zeros(self.max_num_mask, 3 ).float()  # first-dim is 0/1, meaning exist or not
                if len(cur_data['all_parts']) > self.max_num_mask:
                    return None
                if len(cur_data['all_parts']) <= 1:
                    return None
                for i in range(len(part_order)):
                    part_id = part_order[i]
                    if len(cur_data[part_id]['similar']) > self.max_num_similar_parts:
                        return None
                    out[i, :] = torch.from_numpy(cur_data[part_id]['bbox_size']).float()
                out = out.unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'anchor':
                out = torch.zeros(self.max_num_mask, 6, 3 ).float()  # first-dim is 0/1, meaning exist or not
                if len(cur_data['all_parts']) > self.max_num_mask:
                    return None
                if len(cur_data['all_parts']) <= 1:
                    return None
                for i in range(len(part_order)):
                    part_id = part_order[i]
                    if len(cur_data[part_id]['similar']) > self.max_num_similar_parts:
                        return None
                    xyz = cur_data[part_id]['bbox_size']
                    neg_xyz = -1.0 * cur_data[part_id]['bbox_size']
                    out[i, 0, :] = torch.tensor([xyz[0], 0, 0]).float()
                    out[i, 1, :] = torch.tensor([0, xyz[1],0]).float()
                    out[i, 2, :] = torch.tensor([0, 0, xyz[2]]).float()
                    out[i, 3, :] = torch.tensor([neg_xyz[0], 0, 0]).float()
                    out[i, 4, :] = torch.tensor([0, neg_xyz[1], 0]).float()
                    out[i, 5, :] = torch.tensor([0, 0, neg_xyz[2]]).float()

                out = out.unsqueeze(0)
                data_feats = data_feats + (out,)
                

            elif feat == 'sem_one_hot':
                out = torch.zeros( self.max_num_mask, len(self.part_sems)).float()
                if len(cur_data['all_parts']) > self.max_num_mask:
                    return None
                if len(cur_data['all_parts']) <= 1:
                    return None
                for i in range(len(part_order)):
                    part_id = part_order[i]
                    if len(cur_data[part_id]['similar']) > self.max_num_similar_parts:
                        return None
                    out[i, self.part_sem2id[cur_data[part_id]['sem']]] = 1
                out = out.unsqueeze(0)
                data_feats = data_feats + (out,)
        


            elif feat == 'ins_one_hot':
                # instance one hot: shape : max parts allowed  x   max_similar number of  parts
                # out [i, :] is the instance one hot of the parts 
                # [1, 0 .... ], [1, 0, ..... ] [1, 0, ..... ] [0, 1, ..... ]
                out = torch.zeros( self.max_num_mask, self.max_num_similar_parts).float()
                if len(cur_data['all_parts']) > self.max_num_mask:
                    return None
                if len(cur_data['all_parts']) <= 1:
                    return None
                for i in range(len(part_order)):
                    part_id = part_order[i]
                    if len(cur_data[part_id]['similar']) > self.max_num_similar_parts:
                        return None
                    counter = 0
                    for part in sorted(cur_data[part_id]['similar']):

                        ind = part_order.index(part)
                        out[ind, counter] = 1
                        counter += 1
                out = out.unsqueeze(0)
                data_feats = data_feats + (out,)
        
            elif feat == 'shape_id':
                if len(cur_data['all_parts']) > self.max_num_mask:
                    return None
                data_feats = data_feats + (shape_id,)

            elif feat == 'part_id':
                if len(cur_data['all_parts']) > self.max_num_mask:
                    return None
                data_feats = data_feats + (part_order,)

            elif feat == 'view_id':
                if len(cur_data['all_parts']) > self.max_num_mask:
                    return None
                data_feats = data_feats + (view_id,)

            elif feat == 'adj':
                adj_pairs = []
                with open(os.path.join(self.data_dir, self.category, shape_id+'-adj.txt'), 'r') as f:
                    for line in f:
                        l = line.strip()
                        adj_pairs.append((l.split()[0],l.split()[1]))
                out = []
                for pair in adj_pairs:
                    out.append((part_order.index(pair[0]), part_order.index(pair[1])))
                data_feats = data_feats + (out,)
        

            elif feat == 'adj_pt':
                adj_idx_pairs = []
                with open(os.path.join(self.data_dir, self.category, shape_id+'-adj-ptid.txt'), 'r') as f:
                    for line in f:
                        l = line.strip()
                        adj_idx_pairs.append((l.split()[0],l.split()[1],l.split()[2],l.split()[3]))
                out = []
                for i in range(len(adj_idx_pairs)):
                    pair = adj_idx_pairs[i]
                    pts1 = torch.from_numpy(cur_data[pair[0]]['pts']).float()
                    pts2 = torch.from_numpy(cur_data[pair[1]]['pts']).float()
                    t1, rot1 = cur_data_view[pair[0]]['cam_dof'][:3], cur_data_view[pair[0]]['cam_dof'][3:]
                    t2, rot2 = cur_data_view[pair[1]]['cam_dof'][:3], cur_data_view[pair[1]]['cam_dof'][3:]
                    t1 = torch.from_numpy(t1).float(); t2 = torch.from_numpy(t2).float()
                    rot1 = torch.from_numpy(rot1).float(); rot2 = torch.from_numpy(rot2).float()    
                    num_point = 1000
                    pc1 = qrot(rot1.unsqueeze(0).repeat(num_point, 1), pts1) + t1.unsqueeze(0).repeat(num_point, 1)
                    pc2 = qrot(rot2.unsqueeze(0).repeat(num_point, 1), pts2) + t2.unsqueeze(0).repeat(num_point, 1)
                    xyz1 = pc1[int(pair[2])]
                    xyz2 = pc2[int(pair[3])]
                    xyz1 = xyz1.float()
                    xyz2 = xyz2.float()
                    cur_pair_xyz = torch.zeros(2,3)
                    cur_pair_xyz[0] = xyz1; cur_pair_xyz[1] = xyz2
                    out.append(cur_pair_xyz)
                out = torch.stack(out)
                data_feats = data_feats + (out,)


            elif feat == 'total_parts_cnt':
                if len(cur_data['all_parts']) > self.max_num_mask:
                    return None
                if len(cur_data['all_parts']) <= 1:
                    return None
                out = len(cur_data['all_parts'])
                data_feats = data_feats + (out,)

            elif feat == 'similar_parts_cnt':  
                # shape: max_mask  x  2
                # first index is the index of parts without similar parts 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, .... 
                # second index is the number of similar part count:       1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, ....  
                out = torch.zeros(self.max_num_mask, 1, dtype = torch.long)
                if len(cur_data['all_parts']) > self.max_num_mask:
                    return None
                if len(cur_data['all_parts']) <= 1:
                    return None
                total_part_cnt = len(cur_data['all_parts'])
                counter = 0
                for i in range(len(part_order)):
                    part_id = part_order[i]
                    if len(cur_data[part_id]['similar']) > self.max_num_similar_parts:
                        return None
                    out[i, 0] = len(cur_data[part_id]['similar'])
                
                out = out.unsqueeze(0)
                data_feats = data_feats + (out,)
            
            elif feat == 'similar_parts_edge_indices':  
                out = torch.Tensor(equiv_edge_indices).long()
                data_feats = data_feats + (out,)

            elif feat == 'parts_cam_dof':
                out = torch.zeros(self.max_num_mask, 7).float()
                if len(cur_data[part_id]['similar']) > self.max_num_similar_parts:
                    return None
                if len(cur_data['all_parts']) <= 1:
                    return None
                for i in range(len(part_order)):
                    part_id = part_order[i]
                    if len(cur_data[part_id]['similar']) > self.max_num_similar_parts:
                        return None
                    out[i, :] = torch.from_numpy(cur_data_view[part_id]['cam_dof'])
                data_feats = data_feats + (out,)

            else:
                raise ValueError('ERROR: unknown feat type %s!' % feat)

        return data_feats

