# coding:utf-8
import os,glob,time,calendar,timeit, h5py
import numpy as np
import xarray as xr
import netCDF4 as nc
from datetime import datetime, timedelta
import pandas as pd
import torch.utils.data
import concurrent.futures
import multiprocessing

from utils import clean_data

def read_nc(args):
    date, rpath, to_log10 = args
    file_pattern = "AQUA_MODIS." + date.strftime("%Y%m%d") + "*.nc"
    file_path_list = glob.glob(os.path.join(rpath, file_pattern))
    if not file_path_list:
        print(f"File [{rpath.split('/')[-1]}]_[{date.strftime('%Y%m%d')}] not found!")
        return np.full((4320, 8640), np.nan, dtype=np.float16)
    
    file_path = file_path_list[0]
    with xr.open_dataset(file_path) as ds:
        vari = file_path.split(".")[-3]
        data = ds[vari][:].values
        data = np.ma.filled(data, np.nan).astype(np.float16)
    if to_log10:
        data[data <= 0] = np.nan
        result = np.log10(data+1e-5)
        return result
    else:
        return data


class multi_ds(torch.utils.data.Dataset):
    def __init__(self, save_rpath, targ_varis, tasks_extra_fun=None):
        super(multi_ds, self).__init__()
        self.dates = [datetime.strptime(os.path.basename(file).replace('.h5',''), "%Y-%m-%d") for file in glob.glob(os.path.join(save_rpath, '*.h5'))]
#         self.dates = pd.date_range('2015-01-01','2015-02-28')
        self.save_rpath = save_rpath
        self.targ_varis = targ_varis 
        self.tasks_extra_fun = tasks_extra_fun

    def __getitem__(self, index):
        with h5py.File(self.save_rpath+self.dates[index].strftime("%Y-%m-%d")+'.h5', 'r') as hf:
            inpts = hf['inpt'][:].astype(np.float16)
            targs = {vari:hf[vari][:].astype(np.float16) for vari in self.targ_varis}
        cidx = np.random.choice(inpts.shape[0], size=min(40000, inpts.shape[0]), replace=True)
        inpts,targs = inpts[cidx], {vari:value[cidx] for vari,value in targs.items()}
        if self.tasks_extra_fun is not None:
            targs.update(self.tasks_extra_fun(inpts,targs))
        inpts,targs = clean_data(inpts,targs)
        return inpts,targs

    def __len__(self,):
        return len(self.dates)
    
def collate_fn(batchs):
    inpts, targs = [], {item:[] for item in batchs[0][1]}
    for batch in batchs:
        inp, tar = batch
        inpts.append(inp)
        for item,values in tar.items():
            targs[item].append(values)
    inpts = np.concatenate(inpts, axis=0)
    targs =  {item: np.concatenate(values, axis=0) for item,values in targs.items()}
    return inpts, targs



