import os, sys
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from quaternion import qrot

class RealData(data.Dataset):
    def __init__(self, data_dir, data_features):
        # store parameters
        self.data_dir = data_dir
        self.img_size = 224
        self.max_num_similar_parts = 21
        self.max_num_mask = 20
        self.data_features = data_features
        self.num_points = 1000


    def __str__(self):
        strout = 'real_data'
        return strout
    def get_part_count(self):
    	return 57

    def __len__(self):
        return 1

    def __getitem__(self, index):
        
        equiv_edge_indices = [[1, 2], [2, 1], [3, 4], [4, 3], \
        					[5, 6], [5, 7], [5, 8], [6, 5], [6, 7], [6, 8],\
        					[7, 5], [7, 6], [7, 8], [8, 5], [8, 6], [8, 7]]

        # sort part order to put the single part in front and similar parts in the back
        part_order = ['3', '0', '0', '1', '1', '2', '2', '2', '2']
        
        data_feats = ()
        for feat in self.data_features:
            if feat == 'img':
                img_fn = os.path.join(self.data_dir, 'real.png')
                with Image.open(img_fn) as fimg:
                    out = np.array(fimg, dtype=np.float32) / 255
                white_img = np.ones((self.img_size, self.img_size, 3), dtype=np.float32)
                mask = np.tile(out[:, :, 3:4], [1, 1, 3])
            
                out = out[:, :, :3] * mask + white_img * (1 - mask)
                out = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
                data_feats = data_feats + (out,)
            

            elif feat == 'pts':
                out = torch.zeros(self.max_num_mask, self.num_points, 3 ).float()  # first-dim is 0/1, meaning exist or not
                for i in range(len(part_order)):
                    cur_pc = np.load(self.data_dir + 'normed' + part_order[i] +'.npy')
                    out[i, :, :] = torch.from_numpy(cur_pc).float()
                out = out.unsqueeze(0)
                data_feats = data_feats + (out,)


            elif feat == 'ins_one_hot':
                # instance one hot: shape : max parts allowed  x   max_similar number of  parts
                # out [i, :] is the instance one hot of the parts 
                # [1, 0 .... ], [1, 0, ..... ] [1, 0, ..... ] [0, 1, ..... ]
                out = torch.zeros( self.max_num_mask, self.max_num_similar_parts).float()
                out[0, 0] = 1.0
                out[1, 0] = 1.0
                out[2, 1] = 1.0
                out[3, 0] = 1.0
                out[4, 1] = 1.0
                out[5, 0] = 1.0
                out[6, 1] = 1.0
                out[7, 2] = 1.0
                out[8, 3] = 1.0
                out = out.unsqueeze(0)
                data_feats = data_feats + (out,)


            elif feat == 'total_parts_cnt':
                data_feats = data_feats + (9,)

            elif feat == 'similar_parts_cnt':
                out = torch.zeros(self.max_num_mask, 1, dtype=torch.long)
                out[0, 0] = 1
                out[1, 0] = 2
                out[2, 0] = 2
                out[3, 0] = 2
                out[4, 0] = 2
                out[5, 0] = 4
                out[6, 0] = 4
                out[7, 0] = 4
                out[8, 0] = 4
                data_feats = data_feats + (out,)

            elif feat == 'sem_one_hot':
                out = torch.zeros(self.max_num_mask, 57)
                data_feats = data_feats + (out, )

            elif feat == 'similar_parts_edge_indices':
                out = torch.Tensor(equiv_edge_indices).long()
                data_feats = data_feats + (out,)

            else:
            	print('unknown feat !!!!' + feat)

        return data_feats

