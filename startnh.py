import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics
from neuralhydrology.nh_run import start_run
from neuralhydrology.nh_run import eval_run

# Call start_run with the specified configuration file and GPU
config_file = "Sr_Sb_LSTM.yml"
gpu = 1
start_run(config_file=config_file, gpu=gpu)