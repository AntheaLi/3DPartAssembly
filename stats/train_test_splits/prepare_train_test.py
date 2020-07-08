import os
import sys
import json
import numpy as np
from progressbar import ProgressBar

cat = sys.argv[1]
level = int(sys.argv[2])
split = sys.argv[3]

with open('%s.%s.json' % (cat, split), 'r') as fin:
    ids = [d['anno_id'] for d in json.load(fin)]
print(cat, level, split, len(ids))

data_dir = '/orion/u/kaichun/projects/assembly/partnet_assembly_dataset/%s-%d-final' % (cat, level)

def check_pass(idx):
    try:
        d = np.load(os.path.join(data_dir, idx+'.npy'), allow_pickle=True).item()
        assert len(d['all_parts']) <= 20
        assert os.path.exists(os.path.join(data_dir, idx+'-adj-ptid.txt'))
        assert os.path.exists(os.path.join(data_dir, idx))
        for i in range(24):
            os.path.exists(os.path.join(data_dir, idx, '%02d.npy'%i))
        return True
    except:
        return False

with open('%s-%d-%s.txt' % (cat, level, split), 'w') as fout:
    bar = ProgressBar()
    for i in bar(range(len(ids))):
        if check_pass(ids[i]):
            fout.write('%s\n' % ids[i])

