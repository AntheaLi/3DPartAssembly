exp_name="exp-Chair-3-final-model_v2_nosem-mask-C-3-vanilla-2"
epoch_num=305
eval_name="eval_1"


python ./eval_nosem.py \
--exp_name ${exp_name} \
--result_suffix ${eval_name} \
--model_epoch ${epoch_num} \
--visu_batch 1 \
--overwrite
