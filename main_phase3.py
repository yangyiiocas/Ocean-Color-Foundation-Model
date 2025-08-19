# coding: utf-8
import os
import argparse
import time
import numpy as np
import scipy.io as scio
from datetime import datetime
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from model import model_a
from get_phase2_insitu_data import get_tasks2
import config

class training_obj():
    def __init__(self,):
        self.flag = None
        self.train_inpts = None
        self.train_targs = None
        self.test_inpts = None
        self.test_targs = None

def get_training():
    sp = config.Phase3['sp_varis']

    tasks2 = get_tasks2(sp)
    training = {vari:training_obj() for vari in list(tasks2) if vari in sp}

    for p in training:
        idx = np.random.permutation(tasks2[p].length)
        inpts = tasks2[p].sate[idx]
        targs = tasks2[p].targ[idx]

        training[p].flag = "sp"
        test_idx = np.zeros_like(tasks2[p].date, dtype=bool)
        if p == 'pp':
            test_idx = (tasks2[p].date>=datetime(2012,11,15))
        else:
            test_idx = (tasks2[p].date>=datetime(2016,1,1))
        training[p].train_inpts,training[p].test_inpts = inpts,inpts[test_idx]
        training[p].train_targs,training[p].test_targs = targs,targs[test_idx]


        print(f"""variables: {p}
                           train_inputs_shape: {training[p].train_inpts.shape}
                           train_targets_shape: {training[p].train_targs.shape}
                           test_inputs_shape: {training[p].test_inpts.shape}
                           test_targets_shape: {training[p].test_targs.shape}""")

    return training
if __name__ == '__main__':
    cfg_model = config.model
    ckpt = config.Phase3['ckpt']
    device = config.Phase3['device']
    batch_num = config.Phase3['batch_num']
    lr = config.Phase3['lr']
    weight_decay = config.Phase3['weight_decay']
    epochs = config.Phase3['epochs']
    save_model_path = config.Phase3['save_model_path']
    save_mat = config.Phase3['save_mat']
    seed = config.Phase3['seed']

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    ds = get_training()
    print(list(ds))

    model = model_a(tasks=list(ds), **cfg_model).to(device)
    checkpoint = torch.load(ckpt, map_location=device, weights_only=True)
    model.model.load_state_dict(checkpoint['model'],strict=True)
    model.in_Embed.load_state_dict(checkpoint['in_Embed'],strict=True)
    model.out_dEmbed['pp'].load_state_dict(checkpoint['out_dEmbed']['chlor_a'],strict=True)
    
    criterion = nn.MSELoss()

    model.requires_grad_(False)
    model.out_dEmbed.requires_grad_(True)
    optimizer = torch.optim.Adam(model.out_dEmbed.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        sub_in, sub_out = {}, {}
        loss_epoch = []
        for p,data in ds.items():
            cidx = np.random.choice(data.train_inpts.shape[0], size=data.train_inpts.shape[0], replace=False)
            train_inpts = data.train_inpts[cidx]
            train_targs = data.train_targs[cidx]
            sub_in[p] = np.array_split(train_inpts, batch_num, axis=0)
            sub_out[p] = np.array_split(train_targs, batch_num, axis=0)
        for i in range(batch_num):
            loss_batch = []
            optimizer.zero_grad()
            for p in sub_in:
                inpt = sub_in[p][i].copy()
                targ = sub_out[p][i].copy()
                inpt = torch.tensor(inpt,dtype=torch.float32,device=device, requires_grad=False)
                targ = torch.tensor(targ,dtype=torch.float32,device=device, requires_grad=False)

                out = model(inpt)[p]
                loss_graph = criterion(out.view(-1,1), targ.view(-1,1))
                loss_graph.backward(retain_graph=True)
                loss_batch.append(loss_graph.item())
                print('.',end='')
            optimizer.step()
            print(f'train (batch): [{i+1:2d}]/[{batch_num:2d}]  loss: {np.mean(loss_batch):.6}')

        loss_epoch.append(np.mean(loss_batch))
        print(f'train (epoch): [{epoch+1:2d}]/[{epochs:2d}]  loss: {np.mean(loss_epoch):.6}')

        with torch.no_grad():
            for p,data in ds.items():
                if data.test_inpts.size == 0:
                    continue
                inpt = data.test_inpts.copy()
                targ = data.test_targs.copy()
                inpt = torch.tensor(inpt,dtype=torch.float32,device=device, requires_grad=False)
                targ = torch.tensor(targ,dtype=torch.float32,device=device, requires_grad=False)
                out = model(inpt)[p]
                print(f'test (epoch): [{epoch+1:2d}]/[{epochs:2d}] vari: {p}, loss: {criterion(out, targ.view(-1,1)).item():.6f}',end='')
                targ = data.test_targs.reshape(-1,)
                pred = out.detach().cpu().numpy().flatten()
                print(f'       R2: {r2_score(targ, pred) :.4}    MAE:{np.mean(np.abs(targ-pred)) :.4}')




    checkpoint['out_dEmbed'].update({vari:mdl.state_dict() for vari,mdl in model.out_dEmbed.items()})
    torch.save({'model' : model.model.state_dict(),
                'in_Embed' : model.in_Embed.state_dict(),
                'out_dEmbed': checkpoint['out_dEmbed']}, 
               save_model_path)


    # save data
    save_data = {p:{'pred':None, 'targ':None}.copy() for p in ds}
    with torch.no_grad():
        for p,data in ds.items():
            if data.test_inpts.size == 0:
                    continue
            inpt = data.test_inpts.copy()
            targ = data.test_targs.copy()
            inpt = torch.tensor(inpt,dtype=torch.float32,device=device, requires_grad=False)
            targ = torch.tensor(targ,dtype=torch.float32,device=device, requires_grad=False)
            out = model(inpt)[p]
            print(f'test (epoch): [{epoch+1:2d}]/[{epochs:2d}] vari: {p}, loss: {criterion(out, targ.view(-1,1)).item():.6f}',end='')
            targ = data.test_targs.reshape(-1,)
            pred = out.detach().cpu().numpy().reshape(-1,)
            save_data[p]['pred'] = pred 
            save_data[p]['targ'] = targ  
            print(f'       R2: {r2_score(targ, pred) :.4}    MAE:{np.mean(np.abs(targ-pred)) :.4}')

        scio.savemat(save_mat, save_data)
