# python -u -m torch.distributed.run --nproc_per_node=2 main.py \
#     --model mobilenet_v3_small \
#     --epochs 300 \
#     --batch_size 256 \
#     --lr 4e-3 \
#     --update_freq 2 \
#     --model_ema false \
#     --model_ema_eval false \
#     --use_amp true \
#     --data_path /home/DATA/yipin/dataset/imageNet_1k \
#     --output_dir ./checkpoint 

# python -u -m torch.distributed.run --nproc_per_node=2 main.py \
#     --model mobilenet_v3_large \
#     --epochs 300 \
#     --batch_size 256 \
#     --lr 4e-3 \
#     --update_freq 2 \
#     --model_ema false \
#     --model_ema_eval false \
#     --use_amp true \
#     --data_path /home/DATA/yipin/dataset/imageNet_1k \
#     --output_dir ./checkpoint