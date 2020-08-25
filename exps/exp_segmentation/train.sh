data_dir="/orion/u/kaichun/projects/assembly/partnet_assembly_dataset/"


python ./train.py  \
    --exp_suffix mask-size-Tm-try0 \
    --category Table-mixed \
    --device cuda:1 \
    --model_version model_v6_nosem \
    --num_epoch_every_visu 1 \
    --num_batch_every_visu 1 \
    --epochs 100000 \
    --overwrite \
    --data_dir ${data_dir} \
    --lr_decay_every 5000


