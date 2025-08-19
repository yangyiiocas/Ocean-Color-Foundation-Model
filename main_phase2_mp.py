# coding: utf-8
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging,os,shutil,argparse,time,pickle

import numpy as np
import scipy.io as scio
from datetime import datetime
from sklearn.metrics import r2_score
import scipy.io as scio

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import model_a
from get_phase2_insitu_data import get_tasks2
from get_phase2_insitu_data import tasks2Data
import config 

class training_obj():
    def __init__(self,):
        self.flag = None
        self.info = None
        self.inpts = None
        self.targs = None
        self.train_inpts = None
        self.train_targs = None
        self.test_inpts = None
        self.test_targs = None
        self.preds = []
    def cv_sort_by_index(self, index):
        self.train_inpts = np.concatenate([self.inpts[i].copy() for i in range(len(self.inpts)) if i!=index], axis=0)
        self.test_inpts = self.inpts[index].copy()

        self.train_targs = np.concatenate([self.targs[i].copy() for i in range(len(self.targs)) if i!=index], axis=0)
        self.test_targs = self.targs[index].copy()

def get_training(cv_num):
    sp = config.Phase2['sp_varis']
    tasks2 = get_tasks2()
    with open('phase2_insitu_data.pkl','wb') as f:
        pickle.dump(tasks2, f)
    # with open('phase2_insitu_data.pkl','rb') as f:
    #     tasks2 = pickle.load(f)
    training = {vari:training_obj() for vari in list(tasks2) if vari not in sp}
    
    tasks2['spm'].targ = tasks2['spm'].targ * 5.
    tasks2['turbidity'].targ = tasks2['turbidity'].targ * 5.
    
    for p in training:
        training[p].flag = "normal"
        # inpts and targs 
        idx = np.random.permutation(tasks2[p].length)
        training[p].inpts =  np.array_split(tasks2[p].sate[idx].copy(), cv_num, axis=0)
        training[p].targs = np.array_split(tasks2[p].targ[idx].copy(), cv_num, axis=0)

        training[p].info = {'lat' : tasks2[p].lat[idx], 
                            'lon' : tasks2[p].lon[idx],
                            'date' : np.array([d.strftime("%Y-%m-%d %H:%M:%S") for d in tasks2[p].date[idx]]),
                            'depth' :  tasks2[p].depth[idx],
                            'inpt' : tasks2[p].sate[idx].copy(),
                            'targ' : tasks2[p].targ[idx].copy()}

        print(f"""variables: {p}
                           inputs_shape: {training[p].inpts[0].shape[0]*cv_num}
                           targets_shape: {training[p].targs[0].shape[0]*cv_num}""")
    
    return training

def init_model(device):
    cfg_model = config.model
    ckpt = config.Phase2['ckpt']
    model = model_a(tasks=list(ds), 
                    **cfg_model).to(device)

    checkpoint = torch.load(ckpt, map_location=device, weights_only=True)
    model.model.load_state_dict(checkpoint['model'],strict=True)
    model.in_Embed.load_state_dict(checkpoint['in_Embed'],strict=True)
    return model

def sub_training(ds, cv_index, device='cuda',logger=None):
    inner_path = config.Phase2['inner_path']
    epochs = config.Phase2['epochs']
    sub_max = config.Phase2['sub_max']
    lr = config.Phase2['lr']
    weight_decay = config.Phase2['weight_decay']

    if logger is None:
        logging.basicConfig(filename=inner_path + f'log_undefined.log', level=logging.INFO, format='%(asctime)s %(message)s')
        logger = logging.getLogger()
        logger.warn(f'Starting sub_training with index undefined')

    for p in ds:
        ds[p].cv_sort_by_index(cv_index)

    model = init_model(device=device)
    criterion = nn.MSELoss()

    # -----------------------------------------------------------
    # (1) freezing model in_Embed, backbone
    # -----------------------------------------------------------
    model.model.requires_grad_(False)
    model.in_Embed.requires_grad_(False)
    model.out_dEmbed.requires_grad_(True)
    optimizer = torch.optim.Adam(model.out_dEmbed.parameters(), lr=lr[0], weight_decay=weight_decay[0])
    loss_train = []
    for epoch in range(epochs[0]):
        model.train()
        sub_in, sub_out = {}, {}
        for p,data in ds.items():
            cidx = np.random.choice(data.train_inpts.shape[0], size=data.train_inpts.shape[0], replace=False)
            train_inpts = data.train_inpts[cidx]
            train_targs = data.train_targs[cidx]
            batch_num = np.ceil(len(train_targs)/sub_max)
            sub_in[p] = np.array_split(train_inpts, batch_num, axis=0)
            sub_out[p] = np.array_split(train_targs, batch_num, axis=0)
        len_sub = sum([len(v) for _,v in sub_in.items()])

        loss_batch = []
        optimizer.zero_grad()
        for p in sub_in:
            for i in range(len(sub_in[p])):
                inpt = sub_in[p][i].copy()
                targ = sub_out[p][i].copy()
                inpt = torch.tensor(inpt,dtype=torch.float32,device=device, requires_grad=False)
                targ = torch.tensor(targ,dtype=torch.float32,device=device, requires_grad=False)
                out = model(inpt, cho_v=p)[p]
                loss_graph = criterion(out.view(-1,1), targ.view(-1,1))
                (loss_graph/len_sub).backward(retain_graph=True)
                loss_batch.append(loss_graph.item())
        optimizer.step()
        loss_train.append(np.mean(loss_batch))
        logger.info(f'p1 train (batch): (cv: {cv_index})[{epoch+1:2d}]/[{epochs[0]:2d}]  loss: {loss_train[-1]:.6}')


    # -----------------------------------------------------------
    # (2) training model in_Embed, backbone
    # -----------------------------------------------------------
    model.model.requires_grad_(True)
    model.in_Embed.requires_grad_(True)
    model.out_dEmbed.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr[1], weight_decay=weight_decay[1])
    for epoch in range(epochs[1]):
        # ======================= training one epoch =======================
        model.train()
        sub_in, sub_out = {}, {}
        for p,data in ds.items():
            cidx = np.random.choice(data.train_inpts.shape[0], size=data.train_inpts.shape[0], replace=False)
            train_inpts = data.train_inpts[cidx]
            train_targs = data.train_targs[cidx]
            batch_num = np.ceil(len(train_targs)/sub_max)
            sub_in[p] = np.array_split(train_inpts, batch_num, axis=0)
            sub_out[p] = np.array_split(train_targs, batch_num, axis=0)
        len_sub = sum([len(v) for _,v in sub_in.items()])

        loss_batch = []
        optimizer.zero_grad()
        for p in sub_in:
            for i in range(len(sub_in[p])):
                inpt = sub_in[p][i].copy()
                targ = sub_out[p][i].copy()
                inpt = torch.tensor(inpt,dtype=torch.float32,device=device, requires_grad=False)
                targ = torch.tensor(targ,dtype=torch.float32,device=device, requires_grad=False)
                out = model(inpt, cho_v=p)[p]
                loss_graph = criterion(out.view(-1,1), targ.view(-1,1))
                (loss_graph/len_sub).backward(retain_graph=True)
                loss_batch.append(loss_graph.item())
        optimizer.step()
        loss_train.append(np.mean(loss_batch))
        logger.info(f'p2 train (batch): (cv: {cv_index})[{epoch+1:2d}]/[{epochs[1]:2d}]  loss: {loss_train[-1]:.6}')
        # ======================= end train =======================

        if (epoch+1)%50 != 0:
            continue
        model.eval()
        with torch.no_grad():
            for p,data in ds.items():
                inpt = data.test_inpts.copy()
                targ = data.test_targs.copy()
                inpt = torch.tensor(inpt,dtype=torch.float32,device=device, requires_grad=False)
                targ = torch.tensor(targ,dtype=torch.float32,device=device, requires_grad=False)
                out = model(inpt, cho_v=p)[p]
                logger.info(f'test (epoch): (cv: {cv_index})[{epoch+1:2d}]/[{epochs[1]:2d}] vari: {p}, loss: {criterion(out, targ.view(-1,1)).item():.6f}')
                targ = data.test_targs.reshape(-1,)
                pred = out.detach().cpu().numpy().reshape(-1,)
                logger.info(f'       R2: {r2_score(targ, pred) :.4}    MAE:{np.mean(np.abs(targ-pred)) :.4}')

    # -----------------------------------------------------------
    # freezing model in_Embed, backbone
    # -----------------------------------------------------------
    model.model.requires_grad_(False)
    model.in_Embed.requires_grad_(False)
    model.out_dEmbed.requires_grad_(True)
    optimizer = torch.optim.Adam(model.out_dEmbed.parameters(), lr=lr[2], weight_decay=weight_decay[2])
    for epoch in range(epochs[2]):
        model.train()
        sub_in, sub_out = {}, {}
        for p,data in ds.items():
            cidx = np.random.choice(data.train_inpts.shape[0], size=data.train_inpts.shape[0], replace=False)
            train_inpts = data.train_inpts[cidx]
            train_targs = data.train_targs[cidx]
            batch_num = np.ceil(len(train_targs)/sub_max)
            sub_in[p] = np.array_split(train_inpts, batch_num, axis=0)
            sub_out[p] = np.array_split(train_targs, batch_num, axis=0)
        len_sub = sum([len(v) for _,v in sub_in.items()])

        loss_batch = []
        optimizer.zero_grad()
        for p in sub_in:
            for i in range(len(sub_in[p])):
                inpt = sub_in[p][i].copy()
                targ = sub_out[p][i].copy()
                inpt = torch.tensor(inpt,dtype=torch.float32,device=device, requires_grad=False)
                targ = torch.tensor(targ,dtype=torch.float32,device=device, requires_grad=False)
                out = model(inpt, cho_v=p)[p]
                loss_graph = criterion(out.view(-1,1), targ.view(-1,1))
                (loss_graph/len_sub).backward(retain_graph=True)
                loss_batch.append(loss_graph.item())
        optimizer.step()
        loss_train.append(np.mean(loss_batch))
        logger.info(f'p3 train (batch): (cv: {cv_index})[{epoch+1:2d}]/[{epochs[2]:2d}]  loss: {loss_train[-1]:.6}')


    # -----------------------------------------------------------
    # testing
    # -----------------------------------------------------------
    this_preds = {}
    this_loss = 0
    model.eval()
    with torch.no_grad():
        for p,data in ds.items():
            inpt = data.test_inpts.copy()
            targ = data.test_targs.copy()
            inpt = torch.tensor(inpt,dtype=torch.float32,device=device, requires_grad=False)
            targ = torch.tensor(targ,dtype=torch.float32,device=device, requires_grad=False)
            out = model(inpt, cho_v=p)[p]
            this_loss +=criterion(out.view(-1,1), targ.view(-1,1)).item()
            targ = data.test_targs.reshape(-1,)
            pred = out.detach().cpu().numpy().reshape(-1,)
            this_preds[p] = pred
            logger.info(f'<Results> {p:15} (cv: {cv_index}) R2: {r2_score(targ, pred) :.4}    MAE:{np.mean(np.abs(targ-pred)) :.4}')

        torch.save({'model' : model.model.state_dict(),
                    'in_Embed' : model.in_Embed.state_dict(),
                    'out_dEmbed': {vari:mdl.state_dict() for vari,mdl in model.out_dEmbed.items()}}, 
                    inner_path + f'saved_model-split [m2_({cv_index})].pth')
    logger.info(f'\n******<{cv_index}>: <This loss> <{this_loss/len(ds)}>******')
    return this_preds, this_loss, cv_index

def setup_logger(i):
    logger = logging.getLogger(f'logger_{i}')
    logger.setLevel(logging.INFO)
    inner_path = config.Phase2['inner_path']

    if not logger.handlers:
        file_handler = logging.FileHandler(inner_path + f'log_{i}.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(file_handler)
    return logger
def worker(ds, i):
    pred, loss, index = sub_training(ds, i, device=f'cuda:{i}', logger=setup_logger(i))
    return pred, loss, index

if __name__ == '__main__':
    inner_path = config.Phase2['inner_path']
    save_model_path = config.Phase2['save_model_path']
    save_mat = config.Phase2['save_mat']
    cv_num = config.Phase2['cv_num']

    if not os.path.exists(inner_path):
        os.mkdir(inner_path)
    
    # 读取数据
    ds = get_training(cv_num=cv_num)

    # 多进程并行交叉训练
    results = []
    with ProcessPoolExecutor(max_workers=cv_num) as executor:
        futures = [executor.submit(worker, ds, i) for i in range(cv_num)]
        for future in as_completed(futures):
            results.append(future.result())
    
    all_preds = []
    all_bests = []
    index = []
    for pred, best, cv_index in results:
        all_preds.append(pred)
        all_bests.append(best)
        index.append(cv_index)
    
    idx = np.argsort(index)
    get_preds = [all_preds[i].copy() for i in idx]
    get_bests = [all_bests[i] for i in idx]
    
    save_preds = {p: [pred[p] for pred in get_preds] for p in ds}
    shutil.copyfile(inner_path + f'saved_model-split [m2_({np.argmin(get_bests)})].pth', 
                    save_model_path)

    # 保存数据
    save_data = {p: {} for p in ds}
    for p, data in ds.items():
        preds = np.concatenate(save_preds[p].copy(), axis=0)
        targs = np.concatenate(data.targs.copy(), axis=0)
        save_data[p]['pred'] = preds
        save_data[p]['targ'] = targs
        save_data[p]['info'] = data.info
        print(f'{p}    R2: {r2_score(targs, preds):.4f}    MAE: {np.mean(np.abs(targs - preds)):.4f}')
    
    scio.savemat(save_mat, save_data)

    



        

    


