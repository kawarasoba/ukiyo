# !/bin/bash
python inference.py \
    -c='model_effi_b3_sampling' \
    -m='efficientnet_b3' \
    --executed_epoch=0 \
    --is_best_param=True \
    --pseudo_labeling=True \
    --output_confidence=False 

<< local
python inference.py \
    -c='case_name' \
    -m='model_name' \
    --executed_epoch=700 \
    --fix_state_dict=False \
    --nfold=0 \
    --is_final_epoch=False
local