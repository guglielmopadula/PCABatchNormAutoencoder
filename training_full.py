#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:33:45 2023

@author: cyberguli
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
from datawrapper.data import Data
import sys
from models.AE import AE
import copy
import torch
import numpy as np
from pytorch_lightning import Trainer
torch.autograd.set_detect_anomaly(True)
from pytorch_lightning.plugins.environments import SLURMEnvironment
torch.set_float32_matmul_precision('high')
torch.use_deterministic_algorithms(True)
class DisabledSLURMEnvironment(SLURMEnvironment):
    def detect() -> bool:
        return False

    @staticmethod
    def _validate_srun_used() -> None:
        return

    @staticmethod
    def _validate_srun_variables() -> None:
        return


NUM_WORKERS = os.cpu_count()//2
use_cuda=True if torch.cuda.is_available() else False


use_cuda=True if torch.cuda.is_available() else False

use_cuda=False
AVAIL_GPUS=1 if use_cuda else 0

REDUCED_DIMENSION=20
NUM_TRAIN_SAMPLES=100#400
NUM_TEST_SAMPLES=100#200
BATCH_SIZE = 100
LATENT_DIM=5
SMOOTHING_DEGREE=1
DROP_PROB=0.1



data=Data(batch_size=BATCH_SIZE,num_train=NUM_TRAIN_SAMPLES,
          num_test=NUM_TEST_SAMPLES,
          num_workers=NUM_WORKERS,
          reduced_dimension=REDUCED_DIMENSION, 
          data=np.load("data/snapshots_w.npy"),
          use_cuda=use_cuda)
                               

def custom_test(model,data):
    iterator=iter(data.test_dataloader())
    n_batches=data.num_test//data.batch_size
    tot_loss=0
    for i in range(n_batches):
        batch=next(iterator)
        loss=model.test_step(batch,0)
        tot_loss=tot_loss+loss
    tot_loss=tot_loss/n_batches
    return tot_loss



if __name__ == "__main__":
    torch.manual_seed(100)
    np.random.seed(100)
    if data.use_cuda:
        trainer = Trainer(accelerator='gpu', devices=AVAIL_GPUS,max_epochs=500,log_every_n_steps=1,
                                plugins=[DisabledSLURMEnvironment(auto_requeue=False)],
                                )
    else:
        trainer=Trainer(max_epochs=500,log_every_n_steps=1,
                            plugins=[DisabledSLURMEnvironment(auto_requeue=False)],accelerator="cpu"
                            )

    model=AE(data_shape=data.get_reduced_size(),latent_dim=LATENT_DIM,batch_size=data.batch_size,drop_prob=DROP_PROB,pca=data.pca)
    print("Training of AE has started")

    trainer.fit(model, data)

    torch.save(model,"./saved_models/AE.pt")
    data=Data(batch_size=BATCH_SIZE,num_train=NUM_TRAIN_SAMPLES,
          num_test=NUM_TEST_SAMPLES,
          num_workers=NUM_WORKERS,
          reduced_dimension=REDUCED_DIMENSION, 
          data=np.load("data/snapshots_w.npy"),
          use_cuda=False)
                               

    model=torch.load("./saved_models/AE.pt",map_location="cpu")
    model=model.eval()
  
    with torch.no_grad():
        print(custom_test(model,data))


    
    

    
    
    
    
    
    
    
    
    
    
    
