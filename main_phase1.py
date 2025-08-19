# coding: utf-8
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from dataset_psave import multi_ds, collate_fn
from model import model_a
from sdata import tasks, tasks_extra, tasks_extra_fun
from utils import clean_data
import config

if __name__ == '__main__':
    torch.distributed.init_process_group(backend='nccl')
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device("cuda", local_rank)

    save_rpath = config.Phase1['save_rpath']
    batch_size = config.Phase1['batch_size']
    epochs = config.Phase1['epochs']
    lr = config.Phase1['lr']
    weight_decay = config.Phase1['weight_decay']
    sub_batchs = config.Phase1['sub_batchs']
    save_model_path = config.Phase1['save_model_path']
    load_checkpoint = config.Phase1['load_checkpoint']
    
    cfg_model = config.model

    tasks_all = tasks.copy()
    tasks_all.update(tasks_extra())
    print(list(tasks_all))
    model = model_a(tasks=tasks_all, **cfg_model).to(device)
    if load_checkpoint is not None:
        checkpoint = torch.load(load_checkpoint, map_location=device, weights_only=True)
        model.model.load_state_dict(checkpoint['model'],strict=True)
        model.in_Embed.load_state_dict(checkpoint['in_Embed'],strict=True) 
        print('load_checkpoint!')
    model = DDP(model, device_ids=[local_rank])

    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    criterion = nn.MSELoss()

    ds = multi_ds(save_rpath, tasks, tasks_extra_fun=tasks_extra_fun)
    sampler = DistributedSampler(ds)
    train_loaders = DataLoader(dataset=ds, batch_size=batch_size, sampler=sampler, num_workers=4,
                               collate_fn=collate_fn, drop_last=True)

    m = int(len(ds)/(batch_size)/world_size)
    for epoch in range(epochs):
        start_time = time.time()
        for train_i, (inpts_0, targs_0) in enumerate(train_loaders):
            for sub_batch in range(sub_batchs):
                cidx = np.random.choice(inpts_0.shape[0], size=min(4000, inpts_0.shape[0]), replace=False)
                inpts = inpts_0[cidx]
                targs = {vari:value[cidx] for vari, value in targs_0.items()}
            
                # inpts, targs = clean_data(inpts, targs)
                inpts = torch.from_numpy(inpts).to(torch.float32).to(device)
                out = model(inpts)
                optimizer.zero_grad()
                if dist.get_rank() == 0:
                    print(f'local_rank:[{local_rank}] epoch:[{epoch+1:2d}/{epochs:2d}]-i:[{train_i+1:2d}/{m:2d}]-sub_batch:[{sub_batch+1:2d}/{sub_batchs:2d}]')
                loss_a = 0
                for vari, targ in targs.items():    
                    targ = torch.tensor(targ, dtype=torch.float32, requires_grad=False, device=device)
                    loss_graph = criterion(out[vari], targ.view(-1,1))
                    loss_a = loss_a + loss_graph
                    if dist.get_rank() == 0:
                        print(f'\t {vari}:{loss_graph.item():.4f}', end='')
                loss_a.backward()
                optimizer.step()
                if dist.get_rank() == 0:
                    print(f'\n\tmean loss of all variables: ***{loss_a/len(tasks_all):.4f}***  Run time in epoch: {(time.time()-start_time):.1f} s',end="\n\n")
        if dist.get_rank() == 0:
            torch.save({'model' : model.module.model.state_dict(),
                        'in_Embed' : model.module.in_Embed.state_dict(),
                        'out_dEmbed': {vari:mdl.state_dict() for vari,mdl in model.module.out_dEmbed.items()}}, save_model_path)


# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node 2 main.py

#    机器1：
# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node 2 --master_addr hpc-gpu_10_n5 --master_port 29500 main.py
#    机器2：
# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nnodes=2 --node_rank=1 --nproc_per_node 2 --master_addr hpc-gpu_10_n5 --master_port 29500 main.py

# python -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node 2 --master_addr hpc-gpu_7_m7 --master_port 29500 main_phase1.py
# python -m torch.distributed.launch --nnodes=2 --node_rank=1 --nproc_per_node 2 --master_addr hpc-gpu_7_m7 --master_port 29500 main_phase1.py
