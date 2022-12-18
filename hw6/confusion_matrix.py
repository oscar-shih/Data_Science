import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from utils.options import args
import utils.common as utils
import os
import pandas as pd

from utils.options import args
import utils.common as utils
from model import Model

from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
from torch.optim.lr_scheduler import StepLR as StepLR
from transformers import get_cosine_schedule_with_warmup
from dataPreparer import DataPreparation, Data

device = "cuda" if torch.cuda.is_available() else "cpu"
y_pred = [] 
y_true = []   

def valid(args, loader_valid, model):
    losses = utils.AverageMeter()
    acc = utils.AverageMeter()

    criterion = nn.CrossEntropyLoss()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets, datafile) in enumerate(loader_valid, 1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            _, preds = torch.max(logits, 1)
            y_pred.extend(preds.view(-1).detach().cpu().numpy())        
            y_true.extend(targets.view(-1).detach().cpu().numpy())

    return y_pred, y_true

loader = Data(args, data_path=args.src_data_path, label_path=args.src_label_path)
data_loader_valid = loader.loader_valid
model = Model().to(device)
model.load_state_dict(torch.load("./experiment/train/checkpoint/model_best.pt")["state_dict"])
preds, targets = valid(args, data_loader_valid, model)
cf_matrix = confusion_matrix(targets, preds)
per_cls_acc = cf_matrix.diagonal()/cf_matrix.sum(axis=0)                  
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, class_names, class_names)     
plt.figure(figsize = (9,6))
sns.heatmap(df_cm, annot=True)
plt.xlabel("prediction")
plt.ylabel("label (ground truth)")
plt.savefig("confusion_matrix.png")     