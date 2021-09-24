import torch
import random
import os
import json
import glob
import re
import requests
import numpy as np

from pathlib import Path
from datetime import datetime

HOST = "http://localhost:6006/"

def seed_everything(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)

def increment_path(path, exist_ok=False):
    """[summary]

    Args:
        path ([type]): [description]
        exist_ok (bool, optional): [description]. Defaults to False.
    """

    path = Path(path)
    if(path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def send_to_server(log):
    id = log["config"]["id"]
    API = f"api/v1/{id}/trainingProcess"
    url = HOST + API

    headers = {'Content-Type': 'application/json', 'charset': 'UTF-8', 'Accept': '*/*'}

    try:
        res = requests.post(url, headers=headers, data=json.dumps(log, ensure_ascii=False, indent="\t")) # Post Method
    except Exception as e:
        print(e)

def request_state_training(id):

    API = f"api/v1/state/{id}/training"
    url = HOST + API

    headers = {'Content-Type': 'application/json', 'charset': 'UTF-8', 'Accept': '*/*'}
    data = {
        "id":id,
        "state":True,
    }
    try:
        res = requests.post(url, headers=headers, data=json.dumps(data, ensure_ascii=False, indent="\t")) # Post Method
        print(res)
    except Exception as e:
        print(e)
    
def request_state_done(id):

    API = f"api/v1/state/{id}/done"
    url = HOST + API
    headers = {'Content-Type': 'application/json', 'charset': 'UTF-8', 'Accept': '*/*'}
    data = {
        "id":id,
        "state":False,
    }
    try:
        res = requests.post(url, headers=headers, data=json.dumps(data, ensure_ascii=False, indent="\t")) # Post Method
        print(res)
    except Exception as e:
        print(e)

def temp_message(data):
    
    API ="temp"
    url = HOST+API
    headers = {'Content-Type': 'application/json', 'charset': 'UTF-8', 'Accept': '*/*'}
    try:
        res = requests.post(url, headers=headers, data=json.dumps(data, ensure_ascii=False, indent="\t")) # Post Method
    except Exception as e:
        print(e)

def save_training_result(save_dir, log):
    
    now = datetime.now()
    date_time = now.strftime("%Y/%m/%d, %H:%M:%S")
    log["datetime"] = date_time
    with open(os.path.join(save_dir, "configure.json"), "w") as f:
        json.dump(log , f, indent="\t")

def save_inference_result(save_dir, log):
    with open(os.path.join(save_dir, "inference.json"), "w") as f:
        json.dump(log, f, indent="\t")

def get_log_form(id, save_dir, args):
    
    log = {
        "config":{
            "id":id,
            "name":args.model,
            "task":args.task,
            "save_dir":save_dir,
            "data_dir":args.data_dir,
            "dataset":args.dataset,
            "hparam":{
                "batch_size":args.batch_size,
                "epochs":args.epochs,
                "optimizer":args.optimizer,
                "criterion":args.criterion,
                "augmentation":args.augmentation,
                "resize":args.resize,
                "lr":args.lr,
            },
        },

        "train_iter":{
            "epoch":[],
            "acc":[],
            "loss":[],
        },

        "val_iter":{
            "epoch":[],
            "acc":[],
            "loss":[],
        },
    }

    return log

def get_model_files(path):

    model_list = [model for model in os.listdir(path) if not model.startswith('.')]
    model_list = [model for model in model_list if model.endswith('pt')]

    return model_list

def get_best_model_index(model_list):
    
    max_index = -1
    max_acc = 0

    for i in range(len(model_list)):
        name, dataset, task, epoch, acc, loss = model_list[i].split('_')    
        acc = float(acc.replace("%", ""))

        if max_acc < acc:
            max_acc = acc
            max_index = i

    return max_index

def get_json(path):
    json_data = None
    with open(path) as json_file:
        json_data = json.load(json_file)
    return json_data