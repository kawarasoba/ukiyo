# !/bin/bash
python inference_ensemble.py \
    -c='pseudo_label_try' \
    --output_confidence=True \
    --adjust_inference=False

<< local
python inference.py \
    -c='case_name' \
    --common_model_name=''
local