from nni.experiment import Experiment

search_space = {
    'weight_decay': {'_type': 'loguniform', '_value': [0.00001, 0.01]},
    'lr': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
    'momentum': {'_type': 'uniform', '_value': [0, 1]},
}



command = "python -u main.py \
    --model mobilenet_v2 \
    --epochs 100 \
    --batch_size 64 \
    --update_freq 2 \
    --model_ema false \
    --model_ema_eval false \
    --use_amp true \
    --data_set CIFAR \
    --data_path /ssd/dataset/cifar_100 \
    --output_dir ./checkpoint/mobilenetv2/"

experiment = Experiment('local')
experiment.config.trial_command = command
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 2

experiment.run(12355)