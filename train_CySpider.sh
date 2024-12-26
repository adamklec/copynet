source activate copynet

export CUDA_VISIBLE_DEVICES="4"

epochs=100
dataset="CySpider"
python train.py \
    --model_name "baseline_${dataset}_${epochs}e" \
    --epochs $epochs \
    --use_cuda \
    --vocab_limit 15000 \

