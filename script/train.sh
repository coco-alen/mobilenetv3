# /ssd/dataset/imagenet_1k
# /home/DATA/yipin/dataset/imageNet_1k

# python -u -m torch.distributed.run --nproc_per_node=2 main.py \
#     --model mobilenet_v3_small \
#     --epochs 300 \
#     --batch_size 256 \
#     --lr 4e-3 \
#     --update_freq 2 \
#     --model_ema false \
#     --model_ema_eval false \
#     --use_amp true \
#     --data_path /ssd/dataset/imagenet_1k \
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
#     --data_path /ssd/dataset/imagenet_1k \
#     --output_dir ./checkpoint

# python -u main.py \
#     --model mobilenet_v2 \
#     --epochs 450 \
#     --batch_size 256 \
#     --lr 4e-3 \
#     --update_freq 2 \
#     --model_ema false \
#     --model_ema_eval false \
#     --use_amp true \
#     --data_path /ssd/dataset/imagenet_1k \
#     --output_dir ./checkpoint/mobilenetv2/


python -u main.py \
    --model mobilenet_v2 \
    --epochs 450 \
    --batch_size 64 \
    --lr 4e-3 \
    --update_freq 2 \
    --model_ema false \
    --model_ema_eval false \
    --use_amp true \
    --data_set CIFAR \
    --data_path /ssd/dataset/cifar_100 \
    --output_dir ./checkpoint/mobilenetv2/

# python -u main.py \
#     --model mobilenet_v2 \
#     --epochs 100 \
#     --batch_size 512 \
#     --lr 4e-1 \
#     --update_freq 2 \
#     --model_ema false \
#     --model_ema_eval false \
#     --use_amp true \
#     --data_path /ssd/dataset/imagenet_1k \
#     --finetune ./checkpoint/mobilenetv2/mobilenetv2_pytorch_pretrain.pth \
#     --prune ./prune_config/mobilenetv2.py \
#     --output_dir ./checkpoint/mobilenetv2_prune/