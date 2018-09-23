# camera-ready

# this file contains code snippets which I have found (more or less) useful at
# some point during the project. Probably nothing interesting to see here.

import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

model_id = "13_2_2_2"

with open("/home/fregu856/exjobb/training_logs/multitask/model_" + model_id + "/epoch_losses_train.pkl", "rb") as file:
    train_loss = pickle.load(file)

with open("/home/fregu856/exjobb/training_logs/multitask/model_" + model_id + "/epoch_losses_val.pkl", "rb") as file:
    val_loss = pickle.load(file)

print ("train loss min:", np.argmin(np.array(train_loss)), np.min(np.array(train_loss)))

print ("val loss min:", np.argmin(np.array(val_loss)), np.min(np.array(val_loss)))
