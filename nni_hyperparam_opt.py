from nni.experiment import Experiment

search_space = {
    'weight_decay': {'_type': 'loguniform', '_value': [0.00001, 0.01]},
    'lr': {'_type': 'loguniform', '_value': [0.001, 0.3]},
    'momentum': {'_type': 'uniform', '_value': [0, 1]},
}



command = "python -u main.py \
    --model mobilenet_v2 \
    --epochs 5 \
    --batch_size 64 \
    --update_freq 2 \
    --model_ema false \
    --model_ema_eval false \
    --use_amp true \
    --data_set CIFAR \
    --data_path /ssd/dataset/cifar_100 \
    --nni_hyperparam_opt"
    

experiment = Experiment('local')
experiment.config.experiment_name = 'nni_hyperparam_demo'
experiment.config.trial_command = command
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.max_trial_number = 20
experiment.config.trial_concurrency = 2

experiment.run(8080)

input()
experiment.stop()