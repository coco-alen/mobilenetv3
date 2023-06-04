# python main.py \
#     --model mobilenet_v3_large \
#     --data_set IMNET \
#     --data_path /ssd/dataset/imagenet_1k \
#     --finetune ./checkpoint/mobilenetv3/450_act3_mobilenetv3_large.pth \
#     --batch_size 256 \
#     --eval true

# python main.py \
#     --model mobilenet_v3_small \
#     --data_set IMNET \
#     --data_path /ssd/dataset/imagenet_1k \
#     --finetune ./checkpoint/mobilenetv3/300_act3_mobilenetv3_small.pth \
#     --batch_size 256 \
#     --eval true

python main.py \
    --model mobilenet_v2 \
    --data_set IMNET \
    --data_path /ssd/dataset/imagenet_1k \
    --finetune ./checkpoint/mobilenetv2/mobilenetv2_pytorch_pretrain.pth \
    --batch_size 256 \
    --eval true \
    --quantize ./quantize_config/mobilenetv2_8bit.py