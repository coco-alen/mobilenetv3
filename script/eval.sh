# python main.py \
#     --model mobilenet_v3_large \
#     --data_set IMNET \
#     --data_path /ssd/dataset/imagenet_1k \
#     --finetune ./checkpoint/450_act3_mobilenetv3_large.pth \
#     --output_dir ./checkpoint \
#     --batch_size 256 \
#     --eval true

python main.py \
    --model mobilenet_v3_small \
    --data_set IMNET \
    --data_path /ssd/dataset/imagenet_1k \
    --finetune ./checkpoint/300_act3_mobilenetv3_small.pth \
    --output_dir ./checkpoint \
    --batch_size 256 \
    --eval true