# coding: utf-8
import os,glob,time,calendar,timeit, h5py, argparse,json
import numpy as np
import scipy.io as scio
from datetime import datetime,timedelta
from sklearn.metrics import r2_score
import scipy.io as scio
import pandas as pd 

import torch
import torch.nn as nn

from model import model_a

from dataset_psave import read_nc
from sdata import to_log10, tasks, inpt_varis
import config

class _Temp:
    data = []
    this_date = None
    @classmethod
    def renew(cls, date, seq, rpath, inpt_varis, to_log10):
        if (cls.this_date is None) or ((date - cls.this_date).days != 1):
            cls.init(date, seq, rpath, inpt_varis, to_log10)
        else:
            cls.pop(date, seq, rpath, inpt_varis, to_log10)
    @classmethod
    def init(cls, date, seq, rpath, inpt_varis, to_log10):
        cls.data.clear()
        for t in range(seq):
            s1 = np.stack([read_nc((date-timedelta(days=t), rpath+sub, to_log10[vari])).flatten().astype(np.float16) for vari, sub in inpt_varis.items()], axis=1)
            cls.data.append(s1)
        cls.this_date = date
    @classmethod
    def pop(cls, date, seq, rpath, inpt_varis, to_log10):
        cls.data.pop()
        cls.data.insert(0, np.stack([read_nc((date, rpath+sub, to_log10[vari])).flatten().astype(np.float16) for vari, sub in inpt_varis.items()], axis=1))
        cls.this_date = date


def get_inputs(date):
    seq = config.model['seq_len']
    rpath = config.Generate['rpath']
    _Temp.renew(date, seq, rpath, inpt_varis, to_log10)
    inpt = np.stack(_Temp.data,axis=2)

    dx = 1/24
    lat = np.arange(90-dx/2,-90-dx/2,-dx, dtype=np.float16)
    lon = np.arange(-180+dx/2, 180+dx/2,dx, dtype=np.float16)
    dts = [date-timedelta(days=t) for t in range(seq)]
    lat_g, lon_g = np.meshgrid(lat,lon)
    lat_g = np.tile(lat_g.T.reshape(-1,1,1), (1, 1, seq))
    lon_g = np.tile(lon_g.T.reshape(-1,1,1), (1, 1, seq))

    ymd = np.array([t.timetuple().tm_yday / (366 if calendar.isleap(t.year) else 365) for t in dts], dtype=np.float16)
    mth = np.array([t.month / 12. for t in dts],dtype=np.float16)
    day_g = np.tile(ymd.reshape(1, 1, -1), (inpt.shape[0], 1, 1))
    mon_g = np.tile(mth.reshape(1, 1, -1), (inpt.shape[0], 1, 1))

    inpt = np.concatenate([inpt,
                           np.cos(lon_g/180.*np.pi),
                           np.sin(lon_g/180.*np.pi),
                           lat_g/90.,
                           np.sin(day_g*2*np.pi),
                           np.cos(day_g*2*np.pi),
                           np.sin(mon_g*2*np.pi),
                           np.cos(mon_g*2*np.pi)], axis=1)
    return inpt


class app_obj():
    def __init__(self,):
        self.flag = None

def get_app():
    apps = config.Generate['app']
    training = {vari:app_obj() for vari in apps}
    return training

if __name__ == '__main__':
    cfg_model = config.model
    device = config.Generate['device']
    ckpt2 = config.Generate['ckpt2']
    ckpt3 = config.Generate['ckpt3']
    batch_num = config.Generate['batch_num']
    save_rpath = config.Generate['save_rpath']
    bt = config.Generate['bt']
    et = config.Generate['et']

    ds = get_app()
    model = model_a(tasks=list(ds),**cfg_model).to(device)

    checkpoint_m2 = torch.load(ckpt2, map_location=device, weights_only=True)
    checkpoint_m3 = torch.load(ckpt3, map_location=device, weights_only=True)
    model.model.load_state_dict(checkpoint_m2['model'],strict=True)
    model.in_Embed.load_state_dict(checkpoint_m2['in_Embed'],strict=True)
    for vari in model.out_dEmbed:
        if vari in checkpoint_m2['out_dEmbed']:
            model.out_dEmbed[vari].load_state_dict(checkpoint_m2['out_dEmbed'][vari],strict=True)
        else:
            model.out_dEmbed[vari].load_state_dict(checkpoint_m3['out_dEmbed'][vari],strict=True)

            
    for date in pd.date_range(bt,et):
        inpts = get_inputs(date)
        cho_idx = np.ones(inpts.shape[0], dtype=bool)
        ava_tmp = 1.-(np.isnan(inpts)|np.isinf(inpts)).astype(np.float16)
        # inpt_ava = (np.linspace(0.034,1,30, dtype=np.float16).reshape(1,1,-1)*ava_tmp).sum(axis=2).min(axis=1)>0.93
        inpt_ava = ava_tmp.sum(axis=-1).min(axis=-1)>=0.75
        cho_idx[~inpt_ava] = False
        inpt_ava,ava_tmp = None, None
        
        sub_in = np.array_split(inpts[cho_idx,...], batch_num, axis=0)
        preds_ava = {vari:[] for vari in ds}
        preds = {vari:None for vari in ds}
        for i in range(batch_num):
            with torch.no_grad():
                model.eval()
                inpt = sub_in[i].copy()
                inpt = torch.tensor(inpt,dtype=torch.float32,device=device, requires_grad=False)
                out = model(inpt)
                for p in ds:
                    preds_ava[p].append(out[p].detach().cpu().numpy().flatten().astype(np.float16))
            print('.', end='')
        preds_ava = {v:np.concatenate(c, axis=0) for v,c in preds_ava.items()}
        for vari in preds:
            preds[vari] = np.zeros((4320*8640,)).astype(np.float16)*np.nan
            preds[vari][cho_idx] = preds_ava[vari]
        preds = {v:c.reshape(4320, 8640) for v,c in preds.items()}

        input_mask = (1.- (np.isnan(inpts)|np.isinf(inpts))).sum(axis=-1).min(axis=-1).astype(np.float16).reshape(4320, 8640)
        with h5py.File(save_rpath+'OC_'+date.strftime("%Y-%m-%d")+'.h5', 'w') as hf:
            for vari, values in preds.items():
                hf.create_dataset(vari.replace('-','_'), data=values, compression="gzip")
            hf.create_dataset('input_mask', data=input_mask, compression="gzip")
        input_mask = None
        print(f'\n OC_generated/OC_{date.strftime("%Y-%m-%d")}.h5 done')
