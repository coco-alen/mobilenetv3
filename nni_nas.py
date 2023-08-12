from nni.experiment import Experiment

search_space = {
    'expand_ratio1': {'_type': 'choice', '_value': [1,3,6]},
    'channel1' : {'_type': 'choice', '_value': [8, 16, 24]},
    'num1' : {'_type': 'choice', '_value': [0,1,2,3,4]},
    'stride1' : {'_type': 'choice', '_value': [1,2]},
    'type1' : {'_type': 'choice', '_value': ['DwConv', 'Conv','None']},

    'expand_ratio2': {'_type': 'choice', '_value': [1,3,6]},
    'channel2' : {'_type': 'choice', '_value': [24, 32, 40]},
    'num2' : {'_type': 'choice', '_value': [0,1,2,3,4]},
    'stride2' : {'_type': 'choice', '_value': [1,2]},
    'type2' : {'_type': 'choice', '_value': ['DwConv', 'Conv','None']},

    'expand_ratio3': {'_type': 'choice', '_value': [1,3,6]},
    'channel3' : {'_type': 'choice', '_value': [40, 48, 56]},
    'num3' : {'_type': 'choice', '_value': [0,1,2,3,4]},
    'stride3' : {'_type': 'choice', '_value': [1,2]},
    'type3' : {'_type': 'choice', '_value': ['DwConv', 'Conv','None']},

    'expand_ratio4': {'_type': 'choice', '_value': [1,3,6]},
    'channel4' : {'_type': 'choice', '_value': [56, 64, 96]},
    'num4' : {'_type': 'choice', '_value': [0,1,2,3,4]},
    'stride4' : {'_type': 'choice', '_value': [1,2]},
    'type4' : {'_type': 'choice', '_value': ['DwConv', 'Conv','None']},

    'expand_ratio5': {'_type': 'choice', '_value': [1,3,6]},
    'channel5' : {'_type': 'choice', '_value': [96, 128, 144]},
    'num5' : {'_type': 'choice', '_value': [0,1,2,3,4]},
    'stride5' : {'_type': 'choice', '_value': [1,2]},
    'type5' : {'_type': 'choice', '_value': ['DwConv', 'Conv','None']},

    'expand_ratio6': {'_type': 'choice', '_value': [1,3,6]},
    'channel6' : {'_type': 'choice', '_value': [240, 320, 480]},
    'num6' : {'_type': 'choice', '_value': [0,1,2,3,4]},
    'stride6' : {'_type': 'choice', '_value': [1,2]},
    'type6' : {'_type': 'choice', '_value': ['DwConv', 'Conv','None']},
}



command = "python -u main.py \
    --model mobilenet_v2 \
    --epochs 10 \
    --batch_size 64 \
    --lr 0.1 \
    --drop_path 0.05 \
    --weight_decay 1e-3 \
    --warmup_epochs 0 \
    --smoothing 0.1 \
    --momentum 0.2 \
    --update_freq 2 \
    --model_ema false \
    --model_ema_eval false \
    --use_amp true \
    --data_set CIFAR \
    --data_path /ssd/dataset/cifar_100 \
    --nni_NAS"



experiment = Experiment('local')
experiment.config.experiment_name = 'mobilenetv2_cifar100'
experiment.config.trial_command = command
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'Evolution'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.tuner.class_args['population_size'] = 100
experiment.config.max_trial_number = 20
experiment.config.trial_concurrency = 2

experiment.run(8080)

input()
experiment.stop()